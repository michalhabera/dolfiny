#!/usr/bin/env python3

import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

import dolfinx
import ufl
import basix

import dolfiny

import mesh_block3d_gmshapi as mg

# Basic settings
name = "spectral_elasticity"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
dx, dy, dz = 1.0, 1.0, 1.0
nx, ny, nz = 20, 20, 20

# Create the regular mesh of a block with given dimensions
gmsh_model, tdim = mg.mesh_block3d_gmshapi(name, dx, dy, dz, nx, ny, nz, do_quads=False)

# Get mesh and meshtags
mesh, mts = dolfiny.mesh.gmsh_to_dolfin(gmsh_model, tdim)

# Get merged MeshTags for each codimension
subdomains, subdomains_keys = dolfiny.mesh.merge_meshtags(mesh, mts, tdim - 0)
interfaces, interfaces_keys = dolfiny.mesh.merge_meshtags(mesh, mts, tdim - 1)

# Define shorthands for labelled tags
surface_front = interfaces_keys["surface_front"]
surface_back = interfaces_keys["surface_back"]

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains, metadata={"quadrature_degree": 3})
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces, metadata={"quadrature_degree": 3})

# Function spaces
Ue = basix.ufl.element("P", mesh.basix_cell(), 1, rank=1)

Uf = dolfinx.fem.FunctionSpace(mesh, Ue)

# Define functions
u = dolfinx.fem.Function(Uf, name='u')

u_ = dolfinx.fem.Function(Uf, name='u_')  # boundary conditions

δu = ufl.TestFunction(Uf)

# Define state as (ordered) list of functions
m, δm = [u], [δu]

# Boundary conditions
alpha, u2 = 0.25 * np.pi / 8, 0.40


def u_bar(x):
    return np.array([x[2] * (x[0] - (x[0] * np.cos(alpha) - x[1] * np.sin(alpha))),
                     x[2] * (x[1] - (x[0] * np.sin(alpha) + x[1] * np.cos(alpha))),
                     x[2] * u2])


u_.interpolate(u_bar)

# Kinematics
F = ufl.Identity(3) + ufl.grad(u)
C = F.T * F
# C = F.T + F - ufl.Identity(3)  # linearised C
# Spectral decomposition of C
(c0, c1, c2), (E0, E1, E2) = dolfiny.invariants.eigenstate(C)

# Squares of principal stretches
c = ufl.as_vector([c0, c1, c2])
c = ufl.variable(c)
# Variation of squares of principal stretches
δc = dolfiny.expression.derivative(c, m, δm)

# Elasticity parameters
E = dolfinx.fem.Constant(mesh, 10.0)
nu = dolfinx.fem.Constant(mesh, 0.4)
mu = E / (2 * (1 + nu))
la = E * nu / ((1 + nu) * (1 - 2 * nu))


def strain_energy(i1, i2, i3):
    """Strain energy function
    i1, i2, i3: principal invariants of the Cauchy-Green tensor
    """
    # Determinant of configuration gradient F
    J = ufl.sqrt(i3)  # noqa: F841
    #
    # Classical St. Venant-Kirchhoff
    # Ψ = la / 8 * (i1 - 3)**2 + mu / 4 * ((i1 - 3)**2 + 4 * (i1 - 3) - 2 * (i2 - 3))
    # Modified St. Venant-Kirchhoff
    # Ψ = la / 2 * (ufl.ln(J))**2 + mu / 4 * ((i1 - 3)**2 + 4 * (i1 - 3) - 2 * (i2 - 3))
    # Compressible neo-Hooke
    Ψ = mu / 2 * (i1 - 3 - 2 * ufl.ln(J)) + la / 2 * (J - 1)**2
    # Compressible Mooney-Rivlin (beta = 0)
    # Ψ = mu / 4 * (i1 - 3) + mu / 4 * (i2 - 3) - mu * ufl.ln(J) + la / 2 * (J - 1)**2
    #
    return Ψ


# Invariants (based on spectral decomposition of C)
i1, i2, i3 = dolfiny.invariants.invariants_principal(ufl.diag(c))
# i1, i2, i3 = c[0] + c[1] + c[2], c[0] * c[1] + c[0] * c[2] + c[1] * c[2], c[0] * c[1] * c[2]  # easier for UFL?
# Material model (isotropic)
Ψ = strain_energy(i1, i2, i3)
# Stress measure
s = 2 * ufl.diff(Ψ, c)

# Weak form: components (as one-form)
F = 0.5 * ufl.inner(δc, s) * dx  # isotropic material: eigenprojectors of C and S are identical

# Overall form (as list of forms)
F = dolfiny.function.extract_blocks(F, δm)

# Create output xdmf file -- open in Paraview with Xdmf3ReaderT
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
# Write mesh, meshtags
ofile.write_mesh_meshtags(mesh, mts)

# Options for PETSc backend
opts = PETSc.Options("spectral")

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_atol"] = 1.0e-12
opts["snes_rtol"] = 1.0e-08
opts["snes_max_it"] = 10
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"

# Create nonlinear problem: SNES
problem = dolfiny.snesblockproblem.SNESBlockProblem(F, m, prefix="spectral")

# Identify dofs of function spaces associated with tagged interfaces/boundaries
f_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, interfaces, surface_front)
b_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, interfaces, surface_back)

# Set/update boundary conditions
problem.bcs = [
    dolfinx.fem.dirichletbc(u_, f_dofs_Uf),  # u front face
    dolfinx.fem.dirichletbc(u_, b_dofs_Uf),  # u back face
]

# Solve nonlinear problem
m = problem.solve()

# Assert convergence of nonlinear solver
assert problem.snes.getConvergedReason() > 0, "Nonlinear solver did not converge!"

ofile.write_function(u)

ofile.close()
