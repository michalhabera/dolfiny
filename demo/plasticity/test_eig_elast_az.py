#! /usr/bin/env python3

import dolfinx
import dolfinx.io
import dolfiny.interpolation
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import locate_dofs_topological
import ufl
import numpy
import dolfinx.generation
from mpi4py import MPI
from petsc4py import PETSc

import mat3invariants

mesh = dolfinx.generation.UnitCubeMesh(MPI.COMM_WORLD, 20, 20, 20)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "u.xdmf", "w") as file:
    file.write_mesh(mesh)


order = 1
# Function spaces
V = dolfinx.VectorFunctionSpace(mesh, ("CG", order))
Q = dolfinx.FunctionSpace(mesh, ("DG", 0))

bottom_facets = locate_entities_boundary(mesh, 2, lambda x: numpy.isclose(x[2], 0.0))
top_facets = locate_entities_boundary(mesh, 2, lambda x: numpy.isclose(x[2], 1.0))

bottom_dofs = locate_dofs_topological(V, 2, bottom_facets)
top_dofs = locate_dofs_topological(V.sub(2), 2, top_facets)

# Trial and test functions
du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Functions
u = dolfinx.Function(V)

# Integration measure
dx = ufl.dx(mesh)
# dx = ufl.dx(mesh, metadata={"quadrature_degree": order})

# Kinematics
d = 3
Id = ufl.Identity(d)
F = Id + ufl.grad(u)
C = F.T * F
C = ufl.variable(C)

(e0, e1, e2), (E0, E1, E2) = mat3invariants.eigenstate(C)

# Principal stretches
λ0, λ1, λ2 = ufl.sqrt(e0), ufl.sqrt(e1), ufl.sqrt(e2)
λ0, λ1, λ2 = ufl.variable(λ0), ufl.variable(λ1), ufl.variable(λ2)

# Squares of principal stretches
e0, e1, e2 = ufl.variable(e0), ufl.variable(e1), ufl.variable(e2)

# Elasticity parameters
E = dolfinx.Constant(mesh, 10.0)
nu = dolfinx.Constant(mesh, 0.3)  # up to ~0.45 (numerical incompressible limit)
mu = E / (2 * (1 + nu))
la = E * nu / ((1 + nu) * (1 - 2 * nu))


# Strain energy function
def strain_energy(i1, i2, i3):
    """Strain energy function
    i1, i2, i3: principal invariants of the Cauchy-Green tensor
    """
    # Regularisation parameter
    ε = 1.0e-10
    # determinant of configuration gradient F
    J = ufl.sqrt(i3)
    # compressible neo-Hooke (with ε)
    Ψ = mu / 2 * (i1 - 3 - 2 * ufl.ln(J + ε)) + la / 2 * (J - 1)**2
    # compressible Mooney-Rivlin (beta = 0)
    # Ψ = mu / 4 * (i1 - 3) + mu / 4 * (i2 - 3) - mu * ufl.ln(J) + la / 2 * (J - 1)**2 
    # classical St. Venant-Kirchhoff (with polyconvex regulariser)
    # Ψ = la / 8 * (i1 - 3)**2 + mu / 4 * ((i1 - 3)**2 + 4 * (i1 - 3) - 2 * (i2 - 3)) - ε * ufl.ln(J)
    # modified St. Venant-Kirchhoff
    # Ψ = la / 2 * (ufl.ln(J + ε))**2 + mu / 4 * ((i1 - 3)**2 + 4 * (i1 - 3) - 2 * (i2 - 3))
    #
    return Ψ


# Invariants (based on principal stretches)
i1, i2, i3 = mat3invariants.invariants_principal(ufl.diag(ufl.as_vector([e0, e1, e2])))
# i1, i2, i3 = mat3invariants.invariants_principal(ufl.diag(ufl.as_vector([λ0**2, λ1**2, λ2**2])))

# Material model
Ψ = strain_energy(i1, i2, i3)

# Stress measures
s0, s1, s2 = ufl.diff(Ψ, e0), ufl.diff(Ψ, e1), ufl.diff(Ψ, e2)  # PK2
# p0, p1, p2 = ufl.diff(Ψ, λ0), ufl.diff(Ψ, λ1), ufl.diff(Ψ, λ2)  # PK1

# Variation of strain (stretch) measures
de0, de1, de2 = ufl.derivative(e0, u, v), ufl.derivative(e1, u, v), ufl.derivative(e2, u, v)
# dλ0, dλ1, dλ2 = ufl.derivative(λ0, u, v), ufl.derivative(λ1, u, v), ufl.derivative(λ2, u, v)

# Residual
R = 0.5 * (de0 * s0 + de1 * s1 + de2 * s2) * dx
# R = 1.0 * (dλ0 * p0 + dλ1 * p1 + dλ2 * p2) * dx

dC = ufl.derivative(e0 * E0 + e1 * E1 + e2 * E2, u, v)
R = 0.5 * ufl.inner(dC, s0 * E0 + s1 * E1 + s2 * E2) * dx

# Residual (variant with PK2 spectral reconstruction)
# S = s0 * E0 + s1 * E1 + s2 * E2
# dC = ufl.derivative(C, u, v)
# R = 0.5 * ufl.inner(S, dC) * dx

# Uncomment for reference solution
# i1, i2, i3 = mat3invariants.invariants_principal(C)
# Ψ = strain_energy(i1, i2, i3)
# S = 2 * ufl.diff(Ψ, C)
# # C = e0 * E0 + e1 * E1 + e2 * E2
# dC = ufl.derivative(C, u, v)
# R = 0.5 * ufl.inner(S, dC) * dx

# E = 0.5 * (C - Id)
# S = 2 * mu * E + la * ufl.tr(E) * Id
# dE = ufl.derivative(E, u, v)
# R = 1.0 * ufl.inner(S, dE) * dx

# Compute Jacobian
K = -ufl.derivative(R, u, du)

u_bc_bottom = dolfinx.Function(V)
u_bc_top = dolfinx.Function(V)


def bc_top(x):
    values = numpy.zeros_like(x)
    alpha = 0.0 * numpy.pi / 8
    values[0] = x[0] - (x[0] * numpy.cos(alpha) - x[1] * numpy.sin(alpha))
    values[1] = x[1] - (x[0] * numpy.sin(alpha) + x[1] * numpy.cos(alpha))
    values[2] = 0.5
    
    return values


u_bc_top.interpolate(bc_top)

bcs = [dolfinx.fem.DirichletBC(u_bc_bottom, bottom_dofs),
       dolfinx.fem.DirichletBC(u_bc_top.sub(2), top_dofs)]

for i in range(15):

    print(f"---------------------------------------------- newton iteration: i = {i}")

    if i > 0:
        u_bc_top.vector.set(0.0)

    A = dolfinx.fem.assemble_matrix(K, bcs)
    A.assemble()
    print(f"||A|| = {A.norm()}")

    b = dolfinx.fem.assemble_vector(R)
    dolfinx.fem.apply_lifting(b, [K], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

    print(f"||b|| = {b.norm()}")

    ksp = PETSc.KSP()
    ksp.create(MPI.COMM_WORLD)
    opts = PETSc.Options()
    opts["ksp_type"] = "bicg"
    opts["ksp_rtol"] = 1.e-12
    opts["pc_type"] = "bjacobi"
    # opts["ksp_type"] = "preonly"
    # opts["pc_type"] = "lu"
    # opts["pc_factor_mat_solver_type"] = "mumps"

    ksp.setOperators(A)
    x = A.createVecLeft()

    ksp.setFromOptions()
    ksp.solve(b, x)
    print(ksp.getConvergedReason(), ksp.getIterationNumber())
    print(f"||x|| = {x.norm()}")

    u.vector.axpy(1.0, x)
    print(f"||u|| = {u.vector.norm()}")

    psi_vec = dolfinx.fem.assemble_vector(Ψ * ufl.TestFunction(Q) * dx)
    print(f"||Ψ|| = {psi_vec.norm()}")

    # i1_vec = dolfinx.fem.assemble_vector(i1 * ufl.TestFunction(Q) * dx)
    # print(f"||i1|| = {i1_vec.norm()}")
    # i2_vec = dolfinx.fem.assemble_vector(i2 * ufl.TestFunction(Q) * dx)
    # print(f"||i2|| = {i2_vec.norm()}")
    # i3_vec = dolfinx.fem.assemble_vector(i3 * ufl.TestFunction(Q) * dx)
    # print(f"||i3|| = {i3_vec.norm()}")

    b = dolfinx.fem.assemble_vector(R)
    dolfinx.fem.apply_lifting(b, [K], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)
    print(f"||b|| = {b.norm()} (post-solve)")

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "u.xdmf", "a") as file:
        file.write_function(u, t=i)

    if b.norm() < 1.0e-10:
        break

    if numpy.isnan(b.norm()):
        raise RuntimeError("*** NaN ***")