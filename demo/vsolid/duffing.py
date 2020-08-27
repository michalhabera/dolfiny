#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import ufl

import numpy

import dolfiny.odeint
import dolfiny.function
import dolfiny.snesblockproblem


"""
First order nonlinear system of ODEs: (Duffing oscillator, undamped, unforced)

(1) dot v + s = 0
(2) dot s - st(u, v) = 0

with the constitutive law: s(u) = [a + b * u^2] * u
and its rate form: st(u, v) = [a + 3 * b * u^2] * v
and initial conditions: u(t=0) = u_0, v(t=0) = v_0 and s(t=0) = s(u_0)
"""

mesh = dolfinx.generation.UnitIntervalMesh(MPI.COMM_WORLD, 10)

# Problem parameters
a, b = 1, -0.95

# Initial conditions
u_0, v_0 = 1.0, 0.0


def _s(u):
    return (a + b * u**2) * u  # constitutive law


def _st(u, v):
    return (a + 3 * b * u**2) * v  # rate of constitutive law


V = dolfinx.function.FunctionSpace(mesh, ("DG", 0))
S = dolfinx.function.FunctionSpace(mesh, ("DG", 0))

v = dolfinx.function.Function(V, name="v")
s = dolfinx.function.Function(S, name="s")
vt = dolfinx.function.Function(V, name="vt")
st = dolfinx.function.Function(S, name="st")

u = dolfinx.function.Function(V, name="u")  # state u = u0 + ui(v)
ui = dolfinx.function.Function(V, name="ui")  # helper

δv = ufl.TestFunction(V)
δs = ufl.TestFunction(S)

m, mt, δm = [v, s], [vt, st], [δv, δs]

# Set initial conditions
v.vector.set(v_0)  # initial condition: v
vt.vector.set(-_s(u_0))  # initial rate: vt

s.vector.set(_s(u_0))  # initial condition: s
st.vector.set(_st(u_0, v_0))  # initial rate: st

u.vector.set(u_0)  # initial condition: s

v.vector.ghostUpdate()
s.vector.ghostUpdate()
u.vector.ghostUpdate()
vt.vector.ghostUpdate()
st.vector.ghostUpdate()

# Measure
dx = ufl.Measure("dx", domain=mesh)

# Number of time steps
nT = 125

# Global time
t = dolfinx.Constant(mesh, 0.0)

# Time step size
dt = dolfinx.Constant(mesh, 25 / nT)

# Time integrator
odeint = dolfiny.odeint.ODEInt(t=t, dt=dt, x=m, xt=mt)

# Expression for time-integrated quantities
u_expr = u + odeint.integral_dt(odeint.x1[0], odeint.x1t[0], odeint.x0[0], odeint.x0t[0])

# Strong form residuals
r1 = vt + s
r2 = st - _st(u_expr, v)

# Weighted residual (as one-form)
f = δv * r1 * dx + δs * r2 * dx

# Overall form (as one-form)
F = odeint.discretise_in_time(f)
# Overall form (as list of forms)
F = dolfiny.function.extract_blocks(F, δm)

# Options for PETSc backend
opts = PETSc.Options()
opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_atol"] = 1.0e-09
opts["snes_rtol"] = 1.0e-14
opts["snes_stol"] = 1.0e-10

# Create nonlinear problem
problem = dolfiny.snesblockproblem.SNESBlockProblem(F, m)

# Book-keeping of results
v_, vt_ = numpy.zeros(nT + 1), numpy.zeros(nT + 1)
s_, st_ = numpy.zeros(nT + 1), numpy.zeros(nT + 1)
u_ = numpy.zeros(nT + 1)
v_[0], vt_[0] = [w.vector.sum() / w.vector.getSize() for w in [v, vt]]
s_[0], st_[0] = [w.vector.sum() / w.vector.getSize() for w in [s, st]]
u_[0] = u.vector.sum() / u.vector.getSize()

# Process time steps
for time_step in range(1, nT + 1):

    dolfiny.utils.pprint(f"\n+++ Processing time instant = {t.value + dt.value:7.3f} in step {time_step:d}")

    # Stage next time step
    odeint.stage()

    # Solve nonlinear problem
    problem.solve()

    # Assert convergence of nonlinear solver
    assert problem.snes.getConvergedReason() > 0, "Nonlinear solver did not converge!"

    # Update solution states for time integration
    odeint.update()

    # Assert zero residual at t + dt
    assert numpy.isclose(dolfiny.expression.assemble(r1, dx), 0.0, atol=1e-8), "Non-zero residual r1 at (t + dt)!"
    assert numpy.isclose(dolfiny.expression.assemble(r2, dx), 0.0, atol=1e-8), "Non-zero residual r2 at (t + dt)!"

    # Assign time-integrated quantities
    dolfiny.interpolation.interpolate(u_expr, ui)
    dolfiny.interpolation.interpolate(ui, u)

    # Store results
    v_[time_step], vt_[time_step] = [w.vector.sum() / w.vector.getSize() for w in [v, vt]]
    s_[time_step], st_[time_step] = [w.vector.sum() / w.vector.getSize() for w in [s, st]]
    u_[time_step] = u.vector.sum() / u.vector.getSize()


# Compare with reference solution
import sys
sys.path.append("../../test")
from test_odeint import ode_2nd_nonlinear_closed

u_c, v_c, a_c = u_, v_, vt_
u_e, v_e, a_e = ode_2nd_nonlinear_closed(a, b, u_0, nT=nT, dt=dt.value)

if MPI.COMM_WORLD.rank == 0:

    import matplotlib.pyplot

    fig, ax1 = matplotlib.pyplot.subplots()
    ax1.set_title("Duffing oscillator: solution", fontsize=12)
    ax1.set_xlabel(r'time $t$', fontsize=12)
    ax1.set_ylabel(r'solution $u(t)$', fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()
    tt = numpy.linspace(0, dt.value * nT, nT + 1)
    ax1.plot(tt, u_e, color='tab:blue', linestyle='-', linewidth=1.0, label='u_e')
    ax1.plot(tt, u_c, color='tab:blue', linestyle='--', linewidth=1.0, marker='.', label='u_c')
    # ax1.plot(tt, v_e, color='tab:green', linestyle='-', linewidth=1.0, label='v_e')
    # ax1.plot(tt, v_c, color='tab:green', linestyle='--', linewidth=1.0, marker='.', label='v_c')
    # ax1.plot(tt, a_e, color='tab:orange', linestyle='-', linewidth=1.0, label='a_e')
    # ax1.plot(tt, a_c, color='tab:orange', linestyle='--', linewidth=1.0, marker='.', label='a_c')
    ax1.legend(loc='lower left')
    fig.savefig("duffing_solution.pdf")
