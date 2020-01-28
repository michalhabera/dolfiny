from petsc4py import PETSc
import ufl
import numpy
import dolfinx
import dolfinx.cpp
import dolfinx.io

import dolfiny
import dolfiny.restriction
import dolfiny.la
import dolfiny.snesblockproblem


def test_restricted_fs():

    # dS integrals in parallel require shared_facet ghost mode
    if dolfinx.MPI.comm_world.size == 1:
        ghost_mode = dolfinx.cpp.mesh.GhostMode.none
    else:
        ghost_mode = dolfinx.cpp.mesh.GhostMode.shared_facet

    mesh = dolfinx.generation.UnitSquareMesh(dolfinx.MPI.comm_world, 16, 16, ghost_mode=ghost_mode)

    mf = dolfinx.MeshFunction("size_t", mesh, mesh.topology.dim, 0)
    mf.mark(lambda x: numpy.less_equal(x[0], 0.5), 1)
    mf.mark(lambda x: numpy.greater_equal(x[0], 0.5), 2)

    mf_interface = dolfinx.MeshFunction("size_t", mesh, mesh.topology.dim - 1, 0)
    mf_interface.mark(lambda x: numpy.isclose(x[0], 0.5), 1)

    subcells1 = numpy.where(mf.values == 1)[0]
    subcells2 = numpy.where(mf.values == 2)[0]
    interfacecells = numpy.where(mf_interface.values == 1)[0]

    U0 = dolfinx.FunctionSpace(mesh, ("P", 1))
    U1 = dolfinx.FunctionSpace(mesh, ("P", 2))
    L = dolfinx.FunctionSpace(mesh, ("P", 1))

    v0 = ufl.TestFunction(U0)
    v1 = ufl.TestFunction(U1)
    m = ufl.TestFunction(L)

    u0_bc = dolfinx.Function(U0)
    u1_bc = dolfinx.Function(U1)

    dx = ufl.Measure("dx", subdomain_data=mf, domain=mesh)
    dS = ufl.Measure("dS", subdomain_data=mf_interface, domain=mesh)

    w0 = dolfinx.Function(U0, name="w0")
    w1 = dolfinx.Function(U1, name="w1")
    lam = dolfinx.Function(L, name="l")

    b0 = v0 * dx(1)
    b1 = v1 * dx(2)
    b2 = dolfinx.Function(L)("+") * m("+") * dS(1)

    F0 = ufl.inner(ufl.grad(w0), ufl.grad(v0)) * dx(1) + lam("-") * v0("-") * dS(1) - b0
    F1 = ufl.inner(ufl.grad(w1), ufl.grad(v1)) * dx(2) - lam("+") * v1("+") * dS(1) - b1
    F2 = w0("-") * m("-") * dS(1) - w1("+") * m("+") * dS(1) - b2

    bcdofsU0 = dolfinx.fem.locate_dofs_geometrical(U0, lambda x: numpy.isclose(x[0], 0.0))
    bcdofsU1 = dolfinx.fem.locate_dofs_geometrical(U1, lambda x: numpy.isclose(x[0], 1.0))
    bcs = [dolfinx.fem.DirichletBC(u0_bc, bcdofsU0), dolfinx.fem.DirichletBC(u1_bc, bcdofsU1)]

    rdofsU0 = dolfinx.fem.locate_dofs_topological(U0, mesh.topology.dim, subcells1)
    rdofsU1 = dolfinx.fem.locate_dofs_topological(U1, mesh.topology.dim, subcells2)
    rdofsL = dolfinx.fem.locate_dofs_topological(L, mesh.topology.dim - 1, interfacecells)

    owned_sizeU0 = U0.dofmap.index_map.size_local
    owned_sizeU1 = U1.dofmap.index_map.size_local
    owned_sizeL = L.dofmap.index_map.size_local

    rdofsU0 = numpy.extract(rdofsU0 < owned_sizeU0, rdofsU0)
    rdofsU1 = numpy.extract(rdofsU1 < owned_sizeU1, rdofsU1)
    rdofsL = numpy.extract(rdofsL < owned_sizeL, rdofsL)

    r = dolfiny.restriction.Restriction([U0, U1, L], [rdofsU0, rdofsU1, rdofsL])

    opts = PETSc.Options()

    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_rtol"] = 1.0e-08
    opts["snes_max_it"] = 5
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    opts["pc_factor_mat_solver_type"] = "mumps"

    problem = dolfiny.snesblockproblem.SNESBlockProblem([F0, F1, F2], [w0, w1, lam], bcs=bcs, opts=opts, restriction=r)
    s0, s1, s2 = problem.solve()

    # Evaluate the solution -0.5*x*(x-1) at x=0.5
    bb_tree = dolfinx.cpp.geometry.BoundingBoxTree(mesh, 2)
    p = [0.5, 0.5, 0.0]
    cell = dolfinx.cpp.geometry.compute_first_collision(bb_tree, p)
    if cell >= 0:
        value_s0 = s0.eval(p, numpy.asarray(cell))
        value_s1 = s1.eval(p, numpy.asarray(cell))

        assert(numpy.isclose(value_s0[0], 0.125, rtol=1.0e-4))
        assert(numpy.isclose(value_s1[0], 0.125, rtol=1.0e-4))
