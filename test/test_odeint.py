from mpi4py import MPI

import dolfinx
from dolfinx.generation import UnitCubeMesh
from dolfinx.function import Function, FunctionSpace

import ufl
import numpy

import dolfiny.odeint
import dolfiny.function
import dolfiny.snesblockproblem


def ode_1st_order_closed(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1):
    """
    Solve ODE in closed form (analytically, at discrete time instances).

    First order ODE: dot u + a * u = b with initial condition u(t=0) = u_0
    """

    t = numpy.linspace(0, nT * dt, num=nT + 1)

    u = (u_0 - b / a) * numpy.exp(-a * t) + b / a

    return u


def ode_1st_order_eulerf(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1):
    """
    Solve ODE with method "Euler forward".

    First order ODE: dot u + a * u = b with initial condition u(t=0) = u_0
    """

    u = numpy.zeros(nT + 1)

    u[0] = u_0

    for n in range(1, nT + 1):
        u[n] = dt * b + u[n - 1] * (1 - dt * a)

    return u


def ode_1st_order_eulerb(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1):
    """
    Solve ODE with method "Euler backward".

    First order ODE: dot u + a * u = b with initial condition u(t=0) = u_0
    """

    u = numpy.zeros(nT + 1)

    u[0] = u_0

    for n in range(1, nT + 1):
        u[n] = (dt * b + u[n - 1]) / (1 + dt * a)

    return u


def ode_1st_order_crankn(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1):
    """
    Solve ODE with method "Crank Nicolson".

    First order ODE: dot u + a * u = b with initial condition u(t=0) = u_0
    """

    u = numpy.zeros(nT + 1)

    u[0] = u_0

    for n in range(1, nT + 1):
        u[n] = (dt * b + u[n - 1] * (1 - 0.5 * dt * a)) / (1 + 0.5 * dt * a)

    return u


def ode_1st_order_galpha(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1, **kwargs):
    """
    Solve ODE with method "Generalised Alpha".

    First order ODE: dot u + a * u = b with initial condition u(t=0) = u_0
    """

    # Default parameters (Backward-Euler)
    alpha_f = 1.0
    alpha_m = 0.5
    gamma = 0.5

    # Parameters from given rho
    if "rho" in kwargs:
        rho = kwargs["rho"]
        alpha_f = 1.0 / (1.0 + rho)
        alpha_m = 0.5 * (3.0 - rho) / (1.0 + rho)
        gamma = 0.5 + alpha_m - alpha_f

    # Parameters directly
    if "alpha_f" in kwargs and "alpha_m" in kwargs and "gamma" in kwargs:
        alpha_f = kwargs["alpha_f"]
        alpha_m = kwargs["alpha_m"]
        gamma = kwargs["gamma"]

    u = numpy.zeros(nT + 1)
    ut = numpy.zeros(nT + 1)

    u[0] = u_0  # initial value
    ut[0] = b - a * u_0  # exact initial rate of this ODE
    # alternative: approximate unknown initial rate from 2x EB solves + 2nd order FD
    # u_ = ode_1st_order_eulerb(a, b, u_0, nT=2, dt=dt/2)
    # ut[0] = (- 3 * u_[0] + 4 * u_[1] - 1 * u_[2]) / dt

    for n in range(1, nT + 1):
        u[n] = b \
               - ((1 - alpha_m) - alpha_m * (1 - gamma) / gamma) * ut[n - 1] \
               - (-alpha_m / gamma / dt + a * (1 - alpha_f)) * u[n - 1]
        u[n] /= alpha_m / gamma / dt + a * alpha_f

        ut[n] = (u[n] - u[n - 1]) / dt - (1 - gamma) * ut[n - 1]
        ut[n] /= gamma

    return u


# def ode_1st_order_odeint(a=1.0, b=0.0, u_0=1.0, nT=10, **kwargs):
#     """
#     Create 1st order ODE problem and solve with `ODEInt` time integrator.

#     First order ODE: dot u + a * u = b with initial condition u(t=0) = u_0
#     """

#     mesh = UnitCubeMesh(MPI.COMM_WORLD, 1, 1, 1)
#     U = FunctionSpace(mesh, ("DG", 0))

#     u = Function(U, name="u")
#     u0 = Function(U, name="u0")
#     u0t = Function(U, name="u0t")

#     u0.vector.set(u_0)  # initial condition
#     u0t.vector.set(b - a * u_0)  # exact initial rate of this ODE for generalised alpha

#     δu = ufl.TestFunction(U)

#     dx = ufl.Measure("dx", domain=mesh)

#     # Global time
#     time = dolfinx.Constant(mesh, 0.0)

#     # Time integrator
#     odeint = dolfiny.odeint.ODEInt(x=u, x0=u0, x0t=u0t, **kwargs)

#     # Weak form: time-rate-dependent components (as one-form)
#     @odeint.form_hook
#     def g(dudt):
#         g = δu * dudt * dx
#         return g

#     # Weak form: time-rate-independent components (as one-form)
#     @odeint.form_hook
#     def f(u):
#         f = δu * a * u * dx - δu * b * dx
#         return f

#     # Overall form (as one-form)
#     F = odeint.discretise_in_time(g, f)
#     # Overall form (as list of forms)
#     F = dolfiny.function.extract_blocks(F, [δu])

#     # Create problem
#     problem = dolfiny.snesblockproblem.SNESBlockProblem(F, [u])

#     u_avg = numpy.zeros(nT + 1)
#     u_avg[0] = u0.vector.norm(0) / u0.vector.getSize()

#     # Process time steps
#     for time_step in range(1, nT + 1):

#         # Set current time
#         time.value = time_step * odeint.dt

#         # Solve nonlinear problem
#         u, = problem.solve()

#         # Update solution states for time integration
#         odeint.update()

#         u_avg[time_step] = u.vector.norm(0) / u.vector.getSize()

#     return u_avg


def ode_1st_order_odeint(a=1.0, b=0.0, u_0=1.0, nT=10, **kwargs):
    """
    Create 1st order ODE problem and solve with `ODEInt` time integrator.

    First order ODE: dot u + a * u = b with initial condition u(t=0) = u_0
    """

    mesh = UnitCubeMesh(MPI.COMM_WORLD, 1, 1, 1)
    U = FunctionSpace(mesh, ("DG", 0))

    u = Function(U, name="u")
    ut = Function(U, name="ut")

    u0 = Function(U, name="u0")
    u0t = Function(U, name="u0t")

    u0.vector.set(u_0)  # initial condition
    u0t.vector.set(b - a * u_0)  # exact initial rate of this ODE for generalised alpha

    δu = ufl.TestFunction(U)

    dx = ufl.Measure("dx", domain=mesh)

    # Global time
    time = dolfinx.Constant(mesh, 0.0)

    # Time step size
    dt = dolfinx.Constant(mesh, kwargs["dt"])

    # Time integrator
    odeint = dolfiny.odeint.ODEInt(t=time, dt=dt, x=u, xt=ut, x0=u0, x0t=u0t)  # TODO: **kwargs

    # Weak form (as one-form)
    f = δu * ut * dx + δu * a * u * dx - δu * b * dx

    # Overall form (as one-form)
    F = odeint.discretise_in_time(f, u, ut)
    # Overall form (as list of forms)
    F = dolfiny.function.extract_blocks(F, [δu])

    # Create problem
    problem = dolfiny.snesblockproblem.SNESBlockProblem(F, [u])

    u_avg = numpy.zeros(nT + 1)
    u_avg[0] = u0.vector.norm(0) / u0.vector.getSize()

    # Process time steps
    for time_step in range(1, nT + 1):

        # Stage next time step
        odeint.stage()

        # Solve nonlinear problem
        u, = problem.solve()

        # Update solution states for time integration
        odeint.update()

        u_avg[time_step] = u.vector.norm(0) / u.vector.getSize()

    return u_avg


def test_odeint_highlevel():

    # (1) Check wrt hand-coded results

    # Euler forward
    u_computed = ode_1st_order_odeint(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1, alpha_f=0.0, alpha_m=0.5, gamma=0.5)
    u_expected = ode_1st_order_eulerf(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1)
    assert(numpy.isclose(u_computed, u_expected, rtol=1.0e-12).all())

    return

    # Euler backward
    u_computed = ode_1st_order_odeint(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1, alpha_f=1.0, alpha_m=0.5, gamma=0.5)
    u_expected = ode_1st_order_eulerb(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1)
    assert(numpy.isclose(u_computed, u_expected, rtol=1.0e-12).all())

    # Crank-Nicolson
    u_computed = ode_1st_order_odeint(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1, rho=1.0)
    u_expected = ode_1st_order_crankn(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1)
    assert(numpy.isclose(u_computed, u_expected, rtol=1.0e-12).all())

    # Generalised Alpha, rho = 0.5
    u_computed = ode_1st_order_odeint(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1, rho=0.5)
    u_expected = ode_1st_order_galpha(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1, rho=0.5)
    assert(numpy.isclose(u_computed, u_expected, rtol=1.0e-12).all())

    # (2) Check wrt convergence order

    # Generalised alpha
    mi = {'euler_forward': {'param': {'alpha_f': 0.0, 'alpha_m': 0.5, 'gamma': 0.5}, 'order': 1},
          'euler_backward': {'param': {'alpha_f': 1.0, 'alpha_m': 0.5, 'gamma': 0.5}, 'order': 1},
          'crank_nicolson': {'param': {'alpha_f': 0.5, 'alpha_m': 0.5, 'gamma': 0.5}, 'order': 2},
          'generalised_alpha_rho_1.0': {'param': {'rho': 1.0}, 'order': 2},
          'generalised_alpha_rho_0.8': {'param': {'rho': 0.8}, 'order': 2},
          'generalised_alpha_rho_0.6': {'param': {'rho': 0.6}, 'order': 2},
          'generalised_alpha_rho_0.4': {'param': {'rho': 0.4}, 'order': 2},
          'generalised_alpha_rho_0.2': {'param': {'rho': 0.2}, 'order': 2},
          'generalised_alpha_rho_0.0': {'param': {'rho': 0.0}, 'order': 2}, }

    for method, info in mi.items():

        l2 = dict.fromkeys((100, 200, 400), 1.0)

        for N in l2.keys():
            u_computed = ode_1st_order_odeint(a=1.0, b=0.5, u_0=1.0, nT=N, dt=10.0 / N, **info['param'])
            u_expected = ode_1st_order_closed(a=1.0, b=0.5, u_0=1.0, nT=N, dt=10.0 / N)
            l2[N] = numpy.linalg.norm(u_computed - u_expected, 2) / numpy.linalg.norm(u_expected, 2)

        x = numpy.log(numpy.fromiter(l2.keys(), dtype=float))
        y = numpy.log(numpy.fromiter(l2.values(), dtype=float))
        A = numpy.vstack([x, numpy.ones(len(x))]).T
        m = numpy.linalg.lstsq(A, y, rcond=None)[0][0]

        print(f"{method:s}: measured convergence rate = {numpy.abs(m):3.2f}")
        print(l2)

        assert(numpy.isclose(numpy.abs(m), info['order'], rtol=1.0e-1))
