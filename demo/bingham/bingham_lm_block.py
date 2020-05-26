#!/usr/bin/env python3

import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

import dolfinx
import ufl

import dolfiny.io
import dolfiny.mesh
import dolfiny.utils
import dolfiny.odeint
import dolfiny.function
import dolfiny.restriction
import dolfiny.snesblockproblem

import mesh_annulus_gmshapi as mg

# Basic settings
name = "bingham_lm_block"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
iR = 1.0
oR = 2.0
nR = 10 * 4
nT = 7 * 4
x0 = 0.0
y0 = 0.0

# Create the regular mesh of an annulus with given dimensions
gmsh_model, tdim = mg.mesh_annulus_gmshapi(name, iR, oR, nR, nT, x0, y0, do_quads=False)

# # Create the regular mesh of an annulus with given dimensions and save as msh, then read into gmsh model
# mg.mesh_annulus_gmshapi(name, iR, oR, nR, nT, x0, y0, do_quads=False, msh_file=f"{name}.msh")
# gmsh_model, tdim = dolfiny.mesh.msh_to_gmsh(f"{name}.msh")

# Get mesh and meshtags
mesh, mts = dolfiny.mesh.gmsh_to_dolfin(gmsh_model, tdim, prune_z=True)

# # Write mesh and meshtags to file
# with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w") as ofile:
#     ofile.write_mesh_meshtags(mesh, mts)

# # Read mesh and meshtags from file
# with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "r") as ifile:
#     mesh, mts = ifile.read_mesh_meshtags()

# Get merged MeshTags for each codimension
subdomains, subdomains_keys = dolfiny.mesh.merge_meshtags(mts, tdim - 0)
interfaces, interfaces_keys = dolfiny.mesh.merge_meshtags(mts, tdim - 1)

# Define shorthands for labelled tags
ring_inner = interfaces_keys["ring_inner"]
ring_outer = interfaces_keys["ring_outer"]
domain = subdomains_keys["domain"]

# Fluid material parameters
rho = dolfinx.Constant(mesh, 2.0)  # [kg/m^3]
mu = dolfinx.Constant(mesh, 1.0)  # [kg/m/s]
tau_zero = dolfinx.Constant(mesh, 0.2)  # [kg/m/s^2]
tau_zero_regularisation = dolfinx.Constant(mesh, 1.e-3)  # [-]

# Max inner ring velocity
v_inner_max = 0.1  # [m/s]
# Normal and tangential velocity at inner ring
v_n = dolfinx.Constant(mesh, 0.0)  # [m/s]
v_t = dolfinx.Constant(mesh, 0.0)  # [m/s] -- value set/updated in analysis

# Global time
time = dolfinx.Constant(mesh, 0.0)  # [s]
# Time step size
dt = 0.1  # [s]
# Number of time steps
nT = 40

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains, metadata={"quadrature_degree": 4})
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces, metadata={"quadrature_degree": 4})


# Inner ring velocity
def v_inner_(t=0.0, vt=v_inner_max, g=5, a=1, b=3):
    return vt * 0.25 * (np.tanh(g * (t - a)) + 1) * (np.tanh(-g * (t - b)) + 1)


# Function spaces
Ve = ufl.VectorElement("CG", mesh.ufl_cell(), 2)
Pe = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
Le = ufl.FiniteElement("CG", mesh.ufl_cell(), 2)

V = dolfinx.FunctionSpace(mesh, Ve)
P = dolfinx.FunctionSpace(mesh, Pe)
N = dolfinx.FunctionSpace(mesh, Le)
T = dolfinx.FunctionSpace(mesh, Le)

# Define functions
v = dolfinx.Function(V, name="v")
p = dolfinx.Function(P, name="p")
n = dolfinx.Function(N, name="n")
t = dolfinx.Function(T, name="t")

δv = ufl.TestFunction(V)
δp = ufl.TestFunction(P)
δn = ufl.TestFunction(N)
δt = ufl.TestFunction(T)

# Create (zero) initial conditions
v0, v0t = [dolfinx.Function(V)] * 2
p0, p0t = [dolfinx.Function(P)] * 2
n0, n0t = [dolfinx.Function(N)] * 2
t0, t0t = [dolfinx.Function(T)] * 2

# Define state as (ordered) list of functions
m = [v, p, n, t]
m0 = [v0, p0, n0, t0]
m0t = [v0t, p0t, n0t, t0t]
δm = [δv, δp, δn, δt]

# Create other functions
v_vector_o = dolfinx.Function(V)
p_scalar_i = dolfinx.Function(P)

# Set up restriction
rdofsV = dolfiny.mesh.locate_dofs_topological(V, subdomains, domain)
rdofsP = dolfiny.mesh.locate_dofs_topological(P, subdomains, domain)
rdofsN = dolfiny.mesh.locate_dofs_topological(N, interfaces, ring_inner)
rdofsT = dolfiny.mesh.locate_dofs_topological(T, interfaces, ring_inner)
restrc = dolfiny.restriction.Restriction([V, P, N, T], [rdofsV, rdofsP, rdofsN, rdofsT])

# Time integrator
odeint = dolfiny.odeint.ODEInt(dt=dt, x=m, x0=m0, x0t=m0t)


def D(v):
    """Rate of strain as function of v (velocity)."""
    return 0.5 * (ufl.grad(v).T + ufl.grad(v))


def J2(A):
    """Second (main) invariant J2 = (I_1)^2 - 2*(I_2) with I_1, I_2 principal invariants."""
    return 0.5 * ufl.inner(A, A)


def rJ2(A):
    """Square root of J2."""
    return ufl.sqrt(J2(A) + np.finfo(np.float).eps)  # eps for AD


def T(v, p):
    """Constitutive relation for Bingham - Cauchy stress as a function of velocity and pressure."""
    # Deviatoric strain rate
    D_ = ufl.dev(D(v))  # == D(v) if div(v)=0
    # Second invariant
    rJ2_ = rJ2(D_)
    # Regularisation
    mu_effective = mu + tau_zero * 1.0 / (2. * (rJ2_ + tau_zero_regularisation))
    # Cauchy stress
    T = -p * ufl.Identity(2) + 2.0 * mu_effective * D_
    return T


# Weak form: time-rate-dependent components (as one-form)
@odeint.form_hook
def g(dmdt):
    dvdt, _, _, _ = dmdt
    g = ufl.inner(δv, rho * dvdt) * dx
    return g


# Weak form: time-rate-independent components (as one-form)
@odeint.form_hook
def f(m):
    v, p, n, t = m

    n_vec = ufl.FacetNormal(mesh)  # outward unit normal vector
    t_vec = ufl.as_vector([n_vec[1], -n_vec[0]])  # tangent 2D

    f = ufl.inner(δv, rho * ufl.grad(v) * v) * dx \
        + ufl.inner(ufl.grad(δv), T(v, p)) * dx \
        + ufl.inner(δp, ufl.div(v)) * dx \
        - ufl.inner(δv, n_vec) * n * ds(ring_inner) \
        - ufl.inner(δv, t_vec) * t * ds(ring_inner) \
        - δn * (v_n - ufl.inner(v, n_vec)) * ds(ring_inner) \
        - δt * (v_t - ufl.inner(v, t_vec)) * ds(ring_inner)
    return f


# Overall form (as one-form)
F = odeint.discretise_in_time(g, f)
# Overall form (as list of forms)
F = dolfiny.function.extract_blocks(F, δm)

# Create output xdmf file -- open in Paraview with Xdmf3ReaderT
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
# Write mesh, meshtags
ofile.write_mesh_meshtags(mesh, mts)

# Options for PETSc backend
opts = PETSc.Options()

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_rtol"] = 1.0e-08
opts["snes_max_it"] = 12
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"
opts["mat_mumps_icntl_14"] = 500
opts["mat_mumps_icntl_24"] = 1

# Create nonlinear problem: SNES
problem = dolfiny.snesblockproblem.SNESBlockProblem(F, m, opts=opts, restriction=restrc)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
ring_outer_dofs_V = dolfiny.mesh.locate_dofs_topological(V, interfaces, ring_outer)
ring_inner_dofs_P = dolfiny.mesh.locate_dofs_topological(P, interfaces, ring_outer)

# Process time steps
for time_step in range(nT + 1):

    # Set current time
    time.value = time_step * odeint.dt

    # Update functions (taking up time.value)
    v_t.value = v_inner_(t=time.value)

    dolfiny.utils.pprint(f"\n+++ Processing time instant = {time.value:7.3f} in step {time_step:d}")

    # Set/update boundary conditions
    problem.bcs = [
        dolfinx.fem.DirichletBC(v_vector_o, ring_outer_dofs_V),  # velocity ring_outer
        dolfinx.fem.DirichletBC(p_scalar_i, ring_inner_dofs_P),  # pressure ring_inner
    ]

    # Solve nonlinear problem
    m = problem.solve()

    # Assert convergence of nonlinear solver
    assert problem.snes.getConvergedReason() > 0, "Nonlinear solver did not converge!"

    # Extract solution
    v_, p_, n_, t_ = m

    # Write output
    ofile.write_function(v_, time.value)
    ofile.write_function(p_, time.value)
    ofile.write_function(n_, time.value)
    ofile.write_function(t_, time.value)

    # Update solution states for time integration
    m0, m0t = odeint.update()

ofile.close()
