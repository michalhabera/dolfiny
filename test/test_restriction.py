from petsc4py import PETSc
import ufl
import numpy
import dolfin
import dolfin.cpp
import dolfin.io
import pytest

import dolfiny
import dolfiny.restriction
import dolfiny.la
import dolfiny.snesblockproblem

skip_in_parallel = pytest.mark.skipif(
    dolfin.MPI.size(dolfin.MPI.comm_world) > 1,
    reason="This test should only be run in serial.")


@skip_in_parallel
def test_restricted_fs():
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 10, 10)

    mf = dolfin.MeshFunction("size_t", mesh, mesh.topology.dim, 0)
    mf.mark(lambda x: numpy.less_equal(x[0], 0.5), 1)
    mf.mark(lambda x: numpy.greater_equal(x[0], 0.5), 2)

    mf_interface = dolfin.MeshFunction("size_t", mesh, mesh.topology.dim - 1, 0)
    mf_interface.mark(lambda x: numpy.isclose(x[0], 0.5), 1)

    subcells1 = numpy.where(mf.values == 1)[0]
    subcells2 = numpy.where(mf.values == 2)[0]
    interfacecells = numpy.where(mf_interface.values == 1)[0]

    U0 = dolfin.FunctionSpace(mesh, ("P", 1))
    U1 = dolfin.FunctionSpace(mesh, ("P", 1))
    L = dolfin.FunctionSpace(mesh, ("P", 1))

    u0, v0 = ufl.TrialFunction(U0), ufl.TestFunction(U0)
    u1, v1 = ufl.TrialFunction(U1), ufl.TestFunction(U1)
    l, m = ufl.TrialFunction(L), ufl.TestFunction(L)

    u0_bc = dolfin.Function(U0)
    u1_bc = dolfin.Function(U1)

    dx = ufl.Measure("dx", subdomain_data=mf, domain=mesh)
    dS = ufl.Measure("dS", subdomain_data=mf_interface, domain=mesh)

    a00 = ufl.inner(ufl.grad(u0), ufl.grad(v0)) * dx(1)
    a01 = None
    a02 = l("-") * v0("-") * dS(1)
    a10 = None
    a11 = ufl.inner(ufl.grad(u1), ufl.grad(v1)) * dx(2)
    a12 = - l("+") * v1("+") * dS(1)
    a20 = m("-") * u0("-") * dS(1)
    a21 = - m("+") * u1("+") * dS(1)
    a22 = None

    L0 = v0 * dx(1)
    L1 = v1 * dx(2)

    # FIXME: Zero sub-vector via None not yet supported in dolfin-x
    L2 = dolfin.Function(L)("+") * m("+") * dS(1)

    a_block = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    L_block = [L0, L1, L2]

    bcdofsU0 = dolfin.fem.locate_dofs_geometrical(U0, lambda x: numpy.isclose(x[0], 0.0))
    bcdofsU1 = dolfin.fem.locate_dofs_geometrical(U1, lambda x: numpy.isclose(x[0], 1.0))
    bcs = [dolfin.fem.DirichletBC(u0_bc, bcdofsU0), dolfin.fem.DirichletBC(u1_bc, bcdofsU1)]

    rdofsU0 = dolfin.fem.locate_dofs_topological(U0, mesh.topology.dim, subcells1)
    rdofsU1 = dolfin.fem.locate_dofs_topological(U1, mesh.topology.dim, subcells2)
    rdofsL = dolfin.fem.locate_dofs_topological(L, mesh.topology.dim - 1, interfacecells)

    r = dolfiny.restriction.Restriction([U0, U1, L], [rdofsU0, rdofsU1, rdofsL])

    opts = PETSc.Options()
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    opts["pc_factor_mat_solver_type"] = "mumps"

    w0 = dolfin.Function(U0, name="w0")
    w1 = dolfin.Function(U1, name="w1")
    l = dolfin.Function(L, name="l")

    problem = dolfiny.snesblockproblem.SNESBlockProblem(
        L_block, [w0, w1, l], bcs=bcs, J_form=a_block, opts=opts, restriction=r)

    w0, w1, l = problem.solve()
