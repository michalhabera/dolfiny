import os

import numpy

import dolfinx
from dolfinx.mesh import locate_entities_geometrical
import dolfinx.cpp
import dolfinx.io
import dolfiny
import dolfiny.la
import dolfiny.mesh
import dolfiny.restriction
import dolfiny.snesblockproblem
import pytest
import ufl
from mpi4py import MPI
from petsc4py import PETSc

skip_in_parallel = pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1,
    reason="This test should only be run in serial.")


@skip_in_parallel
def test_coupled_poisson():
    # dS integrals in parallel require shared_facet ghost mode
    if MPI.COMM_WORLD.size == 1:
        ghost_mode = dolfinx.cpp.mesh.GhostMode.none
    else:
        ghost_mode = dolfinx.cpp.mesh.GhostMode.shared_facet

    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 16, 16, ghost_mode=ghost_mode)
    mesh.create_connectivity_all()

    left_half = locate_entities_geometrical(mesh, mesh.topology.dim, lambda x: numpy.less_equal(x[0], 0.5))
    right_half = locate_entities_geometrical(mesh, mesh.topology.dim, lambda x: numpy.greater_equal(x[0], 0.5))

    left_values = numpy.full(left_half.shape, 1, dtype=numpy.intc)
    right_values = numpy.full(right_half.shape, 2, dtype=numpy.intc)

    mt = dolfinx.mesh.MeshTags(mesh, mesh.topology.dim, numpy.hstack(
        (left_half, right_half)), numpy.hstack((left_values, right_values)))

    interface_facets = dolfinx.mesh.locate_entities_geometrical(
        mesh, mesh.topology.dim - 1, lambda x: numpy.isclose(x[0], 0.5))
    mt_interface = dolfinx.mesh.MeshTags(mesh, mesh.topology.dim - 1, interface_facets, 1)

    U0 = dolfinx.FunctionSpace(mesh, ("P", 1))
    U1 = dolfinx.FunctionSpace(mesh, ("P", 2))
    L = dolfinx.FunctionSpace(mesh, ("P", 1))

    v0 = ufl.TestFunction(U0)
    v1 = ufl.TestFunction(U1)
    m = ufl.TestFunction(L)

    u0_bc = dolfinx.Function(U0)
    u1_bc = dolfinx.Function(U1)

    dx = ufl.Measure("dx", subdomain_data=mt, domain=mesh)
    dS = ufl.Measure("dS", subdomain_data=mt_interface, domain=mesh)

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

    rdofsU0 = dolfinx.fem.locate_dofs_topological(U0, mesh.topology.dim, left_half)
    rdofsU1 = dolfinx.fem.locate_dofs_topological(U1, mesh.topology.dim, right_half)
    rdofsL = dolfinx.fem.locate_dofs_topological(L, mesh.topology.dim - 1, interface_facets)

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

    assert problem.snes.getConvergedReason() > 0
    assert problem.snes.getIterationNumber() == 1

    # Evaluate the solution -0.5*x*(x-1) at x=0.5
    bb_tree = dolfinx.cpp.geometry.BoundingBoxTree(mesh, 2)
    p = [0.5, 0.5, 0.0]
    cell = dolfinx.cpp.geometry.compute_first_collision(bb_tree, p)
    if cell >= 0:
        value_s0 = s0.eval(p, numpy.asarray(cell))
        value_s1 = s1.eval(p, numpy.asarray(cell))

        assert(numpy.isclose(value_s0[0], 0.125, rtol=1.0e-4))
        assert(numpy.isclose(value_s1[0], 0.125, rtol=1.0e-4))


@skip_in_parallel
def test_sloped_stokes():

    path = os.path.dirname(os.path.realpath(__file__))

    # Read mesh, subdomains and boundaries
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                             os.path.join(path, "data", "sloped_triangle_mesh.xdmf"), "r") as infile:
        mesh = infile.read_mesh(name="Grid")
        mesh.create_connectivity_all()

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, os.path.join(path, "data", "sloped_line_mvc.xdmf"), "r") as infile:
        boundaries = infile.read_meshtags(mesh, name="Grid")

    V = dolfinx.VectorFunctionSpace(mesh, ("P", 2))
    P = dolfinx.FunctionSpace(mesh, ("P", 1))
    L = dolfinx.FunctionSpace(mesh, ("P", 1))

    u = dolfinx.Function(V, name="u")
    v = ufl.TestFunction(V)

    p = dolfinx.Function(P, name="p")
    q = ufl.TestFunction(P)

    lam = dolfinx.Function(L, name="l")
    m = ufl.TestFunction(L)

    n = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", subdomain_data=boundaries, domain=mesh)

    F0 = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(mesh) \
        - p * ufl.div(v) * ufl.dx(mesh) + lam * ufl.inner(v, n) * ds(1)
    F1 = ufl.div(u) * q * ufl.dx(mesh)
    F2 = ufl.inner(u, n) * m * ds(1)

    u_bc = dolfinx.Function(V)
    bc_facets = numpy.where(boundaries.values == 4)[0]
    bcdofsV = dolfinx.fem.locate_dofs_topological(V, 1, boundaries.indices[bc_facets])

    u_bc_top = dolfinx.Function(V)
    bctop_facets = numpy.where(boundaries.values == 3)[0]
    bcdofstopV = dolfinx.fem.locate_dofs_topological(V, 1, boundaries.indices[bctop_facets])

    def utop(x):
        values = numpy.zeros((2, x.shape[1]))
        values[0] = 1.0
        values[1] = 0.0
        return values

    u_bc_top.interpolate(utop)

    # Find common dofs at the top corners
    intersect = numpy.intersect1d(bcdofsV, bcdofstopV)

    # Remove for consistency
    bcdofstopV = numpy.setdiff1d(bcdofstopV, intersect)

    bcs = [dolfinx.fem.DirichletBC(u_bc, bcdofsV)]
    bcs.append(dolfinx.fem.DirichletBC(u_bc_top, bcdofstopV))

    r_facets = numpy.where(boundaries.values == 1)[0]
    rdofsL = dolfinx.fem.locate_dofs_topological(L, 1, boundaries.indices[r_facets])

    Vsize = V.dofmap.index_map.block_size * (V.dofmap.index_map.size_local)
    Psize = P.dofmap.index_map.block_size * (P.dofmap.index_map.size_local)

    rdofsV = numpy.arange(Vsize, dtype=numpy.int32)
    rdofsP = numpy.arange(Psize, dtype=numpy.int32)

    r = dolfiny.restriction.Restriction([V, P, L], [rdofsV, rdofsP, rdofsL])

    opts = PETSc.Options()

    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_rtol"] = 1.0e-08
    opts["snes_max_it"] = 10
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    opts["pc_factor_mat_solver_type"] = "mumps"
    opts['mat_mumps_icntl_24'] = 1

    problem = dolfiny.snesblockproblem.SNESBlockProblem([F0, F1, F2], [u, p, lam], bcs=bcs, opts=opts, restriction=r)
    s0, s1, s2 = problem.solve()

    assert problem.snes.getConvergedReason() > 0
    assert problem.snes.getIterationNumber() == 1


# @skip_in_parallel
# def test_pipes_stokes():
#     import gmsh

#     gmsh.initialize()
#     gmsh.model.add("test_pipes")
#     geo = gmsh.model.geo

#     p0 = geo.addPoint(0.0, 0.0, 0.0)
#     p1 = geo.addPoint(1.0, 0.0, 0.0)
#     p2 = geo.addPoint(0.0, -1.0, 0.0)
#     p3 = geo.addPoint(1.0, -1.0, 0.0)
#     p4 = geo.addPoint(1.0, -2.0, 0.0)
#     p5 = geo.addPoint(2.0, -2.0, 0.0)
#     p6 = geo.addPoint(2.0, -1.0, 0.0)

#     l0 = geo.addLine(p0, p2)
#     l1 = geo.addCircleArc(p2, p3, p4)
#     l2 = geo.addLine(p4, p5)
#     l3 = geo.addLine(p5, p6)
#     l4 = geo.addLine(p6, p3)
#     l5 = geo.addLine(p3, p1)
#     l6 = geo.addLine(p1, p0)

#     cl = geo.addCurveLoop([l0, l1, l2, l3, l4, l5, l6])
#     s0 = geo.addPlaneSurface([cl])

#     # Bottom
#     gmsh.model.addPhysicalGroup(1, [l0, l1, l2], 1)
#     # Up
#     gmsh.model.addPhysicalGroup(1, [l4, l5], 2)
#     # Inflow
#     gmsh.model.addPhysicalGroup(1, [l6], 3)

#     geo.synchronize()
#     gmsh.model.mesh.generate()

#     gmsh.model.mesh.refine()
#     gmsh.model.mesh.refine()

#     gmsh.model.mesh.generate()

#     mesh, mvcs = dolfiny.mesh.gmsh_to_dolfin(gmsh.model, 2, prune_z=True)

#     with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf") as out:
#         out.write(mesh)

#     V = dolfinx.VectorFunctionSpace(mesh, ("P", 2))
#     P = dolfinx.FunctionSpace(mesh, ("P", 1))
#     L = dolfinx.FunctionSpace(mesh, ("P", 1))

#     u = dolfinx.Function(V, name="u")
#     v = ufl.TestFunction(V)

#     p = dolfinx.Function(P, name="p")
#     q = ufl.TestFunction(P)

#     lam = dolfinx.Function(L, name="l")
#     m = ufl.TestFunction(L)

#     bottom = dolfinx.MeshFunction("size_t", mesh, mvcs[(1, 1)], 0)
#     up = dolfinx.MeshFunction("size_t", mesh, mvcs[(1, 2)], 0)
#     inflow = dolfinx.MeshFunction("size_t", mesh, mvcs[(1, 3)], 0)

#     ds_bot = ufl.Measure("ds", subdomain_data=bottom, domain=mesh)
#     n = ufl.FacetNormal(mesh)

#     F0 = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(mesh) \
#         - p * ufl.div(v) * ufl.dx(mesh) + lam * ufl.inner(v, n) * ds_bot(1)
#     F1 = ufl.div(u) * q * ufl.dx(mesh)
#     F2 = ufl.inner(u, n) * m * ds_bot(1)

#     u_inflow = dolfinx.Function(V)
#     inflow_facets = numpy.where(inflow.values == 3)[0]
#     inflowdofsV = dolfinx.fem.locate_dofs_topological(V, 1, inflow_facets)

#     u_up = dolfinx.Function(V)
#     up_facets = numpy.where(up.values == 2)[0]
#     updofsV = dolfinx.fem.locate_dofs_topological(V, 1, up_facets)

#     def uinflow(x):
#         values = numpy.zeros((2, x.shape[1]))
#         values[0] = 0.0
#         values[1] = -1.0
#         return values

#     u_inflow.interpolate(uinflow)

#     p_bc = dolfinx.Function(P)

#     bcdofsP = dolfinx.fem.locate_dofs_geometrical(
#         P, lambda x: numpy.logical_and(numpy.isclose(x[0], 0.0), numpy.isclose(x[1], 0.0)))

#     # Find common dofs at the top corners
#     intersect = numpy.intersect1d(inflowdofsV, updofsV)

#     # Remove for consistency
#     inflowdofsV = numpy.setdiff1d(inflowdofsV, intersect)

#     bcs = [dolfinx.fem.DirichletBC(u_inflow, inflowdofsV)]
#     bcs.append(dolfinx.fem.DirichletBC(u_up, updofsV))
#     bcs.append(dolfinx.fem.DirichletBC(p_bc, bcdofsP))

#     lagrange_facets = numpy.where(bottom.values == 1)[0]
#     lagrangedofsL = dolfinx.fem.locate_dofs_topological(L, 1, lagrange_facets)

#     Vsize = V.dofmap.index_map.block_size * V.dofmap.index_map.size_local
#     Psize = P.dofmap.index_map.block_size * P.dofmap.index_map.size_local

#     rdofsV = numpy.arange(Vsize, dtype=numpy.int32)
#     rdofsP = numpy.arange(Psize, dtype=numpy.int32)

#     r = dolfiny.restriction.Restriction([V, P, L], [rdofsV, rdofsP, lagrangedofsL])

#     opts = PETSc.Options()

#     opts["snes_type"] = "newtonls"
#     opts["snes_linesearch_type"] = "basic"
#     opts["snes_rtol"] = 1.0e-08
#     opts["snes_max_it"] = 10
#     opts["ksp_type"] = "preonly"
#     opts["pc_type"] = "lu"
#     opts["pc_factor_mat_solver_type"] = "mumps"
#     opts['mat_mumps_icntl_24'] = 1

#     problem = dolfiny.snesblockproblem.SNESBlockProblem([F0, F1, F2], [u, p, lam], bcs=bcs, opts=opts, restriction=r)
#     s0, s1, s2 = problem.solve()

#     assert problem.snes.getConvergedReason() > 0
