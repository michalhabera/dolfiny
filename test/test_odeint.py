from mpi4py import MPI

import dolfinx
from dolfinx.generation import UnitIntervalMesh
from dolfinx.function import Function, FunctionSpace

import ufl
import numpy

import dolfiny.odeint
import dolfiny.function
import dolfiny.snesblockproblem


def ode_1st_linear_closed(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1):
    """
    Solve ODE in closed form (analytically, at discrete time instances).

    First order linear ODE:
    dot u + a * u - b = 0 with initial condition u(t=0) = u_0
    """

    t = numpy.linspace(0, nT * dt, num=nT + 1)

    u = (u_0 - b / a) * numpy.exp(-a * t) + b / a
    ut = -a * (u_0 - b / a) * numpy.exp(-a * t)

    return u, ut


def ode_1st_linear_eulerb(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1):
    """
    Solve ODE with method "Euler backward".

    First order linear ODE:
    dot u + a * u - b = 0 with initial condition u(t=0) = u_0
    """

    u = numpy.zeros(nT + 1)

    u[0] = u_0

    for n in range(1, nT + 1):
        u[n] = (dt * b + u[n - 1]) / (1 + dt * a)

    return u


def ode_1st_linear_crankn(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1):
    """
    Solve ODE with method "Crank-Nicolson".

    First order linear ODE:
    dot u + a * u - b = 0 with initial condition u(t=0) = u_0
    """

    u = numpy.zeros(nT + 1)

    u[0] = u_0

    for n in range(1, nT + 1):
        u[n] = (dt * b + u[n - 1] * (1 - 0.5 * dt * a)) / (1 + 0.5 * dt * a)

    return u


def ode_1st_linear_galpha(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1, **kwargs):
    """
    Solve ODE with method "Generalised Alpha".

    First order linear ODE:
    dot u + a * u - b = 0 with initial condition u(t=0) = u_0
    """

    # Default parameters (Crank-Nicolson)
    alpha_f, alpha_m, gamma = 0.5, 0.5, 0.5

    # Parameters from given rho
    if "rho" in kwargs:
        rho = kwargs["rho"]
        alpha_f = 1.0 / (1.0 + rho)
        alpha_m = 0.5 * (3.0 - rho) / (1.0 + rho)
        gamma = 0.5 + alpha_m - alpha_f

    # Parameters directly
    if "alpha_f" in kwargs and "alpha_m" in kwargs and "gamma" in kwargs:
        alpha_f, alpha_m, gamma = kwargs["alpha_f"], kwargs["alpha_m"], kwargs["gamma"]

    u = numpy.zeros(nT + 1)
    ut = numpy.zeros(nT + 1)

    u[0] = u_0  # initial value
    ut[0] = b - a * u_0  # exact initial rate of this ODE
    # alternative: approximate unknown initial rate from 2x EB solves + 2nd order FD
    # u_ = ode_1st_order_eulerb(a, b, u_0, nT=2, dt=dt/2)
    # ut[0] = (- 3 * u_[0] + 4 * u_[1] - 1 * u_[2]) / dt

    v = numpy.copy(ut)  # auxiliary rate

    for n in range(1, nT + 1):
        u[n] = alpha_f * b \
            - ((1 - alpha_m) - alpha_m * (1 - gamma) / gamma) * v[n - 1] \
            + (1 - alpha_f) * ut[n - 1] \
            - (-alpha_m / gamma / dt) * u[n - 1]
        u[n] /= alpha_m / gamma / dt + a * alpha_f

        v[n] = (u[n] - u[n - 1]) / dt - (1 - gamma) * v[n - 1]
        v[n] /= gamma

        ut[n] = (1 - alpha_m) * v[n - 1] + alpha_m * v[n] - (1 - alpha_f) * ut[n - 1]
        ut[n] /= alpha_f

    return u


def ode_1st_linear_odeint(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1, **kwargs):
    """
    Create 1st order ODE problem and solve with `ODEInt` time integrator.

    First order linear ODE:
    dot u + a * u - b = 0 with initial condition u(t=0) = u_0
    """

    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 10)
    U = FunctionSpace(mesh, ("DG", 0))

    u = Function(U, name="u")
    ut = Function(U, name="ut")

    u.vector.set(u_0)  # initial condition
    ut.vector.set(b - a * u_0)  # exact initial rate of this ODE for generalised alpha

    u.vector.ghostUpdate()
    ut.vector.ghostUpdate()

    δu = ufl.TestFunction(U)

    dx = ufl.Measure("dx", domain=mesh)

    # Global time
    time = dolfinx.Constant(mesh, 0.0)

    # Time step size
    dt = dolfinx.Constant(mesh, dt)

    # Time integrator
    odeint = dolfiny.odeint.ODEInt(t=time, dt=dt, x=u, xt=ut, **kwargs)

    # Resdiual form (as one-form)
    r = ut + a * u - b

    # Weighted residual form (as one-form)
    f = δu * r * dx

    # Overall form (as one-form)
    F = odeint.discretise_in_time(f)
    # Overall form (as list of forms)
    F = dolfiny.function.extract_blocks(F, [δu])

    # Silence SNES monitoring during test
    dolfiny.snesblockproblem.SNESBlockProblem.print_norms = lambda self, it: 1

    # Create problem (although having a linear ODE we use the dolfiny.snesblockproblem API)
    problem = dolfiny.snesblockproblem.SNESBlockProblem(F, [u])

    # Book-keeping of results
    u_, ut_ = numpy.zeros(nT + 1), numpy.zeros(nT + 1)
    u_[0], ut_[0] = [v.vector.sum() / v.vector.getSize() for v in [u, ut]]

    dolfiny.utils.pprint(f"+++ Processing time steps = {nT}")

    # Process time steps
    for time_step in range(1, nT + 1):

        # Stage next time step
        odeint.stage()

        # Solve (linear) problem
        problem.solve()

        # Update solution states for time integration
        odeint.update()

        # Assert zero residual at t + dt
        assert numpy.isclose(dolfiny.expression.assemble(r, dx), 0.0, atol=1e-6), "Non-zero residual at (t + dt)!"

        # Store results
        u_[time_step], ut_[time_step] = [v.vector.sum() / v.vector.getSize() for v in [u, ut]]

    return u_, ut_


def ode_1st_nonlinear_closed(a=1.0, b=1.0, c=1.0, nT=10, dt=0.1):
    """
    Solve ODE in closed form (analytically, at discrete time instances).

    First order nonlinear non-autonomous ODE:
    t * dot u - a * cos(c*t) * u^2 - 2*u - a * b^2 * t^4 * cos(c*t) = 0 with initial condition u(t=1) = 0
    """

    t = numpy.linspace(1, 1 + nT * dt, num=nT + 1)

    z = c * t * numpy.sin(c * t) - c * numpy.sin(c) + numpy.cos(c * t) - numpy.cos(c)
    zt = c**2 * t * numpy.cos(c * t)

    u = b * t**2 * numpy.tan(a * b / c**2 * z)
    ut = 2 * b * t * numpy.tan(a * b / c**2 * z) + a * b**2 / c**2 * t**2 * (numpy.tan((a * b / c**2 * z))**2 + 1) * zt

    return u, ut


def ode_1st_nonlinear_odeint(a=1.0, b=1.0, c=1.0, nT=10, dt=0.1, **kwargs):
    """
    Create 1st order ODE problem and solve with `ODEInt` time integrator.

    First order nonlinear non-autonomous ODE:
    t * dot u - a * cos(c*t) * u^2 - 2 * u - a * b^2 * t^4 * cos(c*t) = 0 with initial condition u(t=1) = 0
    """

    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 10)
    U = FunctionSpace(mesh, ("DG", 0))

    u = Function(U, name="u")
    ut = Function(U, name="ut")

    u.vector.set(0.0)  # initial condition
    ut.vector.set(a * b**2 * numpy.cos(c))  # exact initial rate of this ODE for generalised alpha

    u.vector.ghostUpdate()
    ut.vector.ghostUpdate()

    δu = ufl.TestFunction(U)

    dx = ufl.Measure("dx", domain=mesh)

    # Global time
    t = dolfinx.Constant(mesh, 1.0)

    # Time step size
    dt = dolfinx.Constant(mesh, dt)

    # Time integrator
    odeint = dolfiny.odeint.ODEInt(t=t, dt=dt, x=u, xt=ut, **kwargs)

    # Strong form residual (as one-form)
    r = t * ut - a * ufl.cos(c * t) * u**2 - 2 * u - a * b**2 * t**4 * ufl.cos(c * t)

    # Weighted residual (as one-form)
    f = δu * r * dx

    # Overall form (as one-form)
    F = odeint.discretise_in_time(f)
    # Overall form (as list of forms)
    F = dolfiny.function.extract_blocks(F, [δu])

    # Options for PETSc backend
    from petsc4py import PETSc
    opts = PETSc.Options()
    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_atol"] = 1.0e-09
    opts["snes_rtol"] = 1.0e-12

    # Silence SNES monitoring during test
    dolfiny.snesblockproblem.SNESBlockProblem.print_norms = lambda self, it: 1

    # Create nonlinear problem
    problem = dolfiny.snesblockproblem.SNESBlockProblem(F, [u])

    # Book-keeping of results
    u_, ut_ = numpy.zeros(nT + 1), numpy.zeros(nT + 1)
    u_[0], ut_[0] = [v.vector.sum() / v.vector.getSize() for v in [u, ut]]

    dolfiny.utils.pprint(f"+++ Processing time steps = {nT}")

    # Process time steps
    for time_step in range(1, nT + 1):

        # Stage next time step
        odeint.stage()

        # Solve nonlinear problem
        problem.solve()

        # Assert convergence of nonlinear solver
        assert problem.snes.getConvergedReason() > 0, "Nonlinear solver did not converge!"

        # Update solution states for time integration
        odeint.update()

        # Assert zero residual at t + dt
        assert numpy.isclose(dolfiny.expression.assemble(r, dx), 0.0, atol=1e-6), "Non-zero residual at (t + dt)!"

        # Store results
        u_[time_step], ut_[time_step] = [v.vector.sum() / v.vector.getSize() for v in [u, ut]]

    return u_, ut_


def test_odeint_linear():

    # *** Check wrt hand-coded results

    # Euler backward
    computed = ode_1st_linear_odeint(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1, alpha_f=1.0, alpha_m=0.5, gamma=0.5)
    expected = ode_1st_linear_eulerb(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1)
    assert(numpy.isclose(computed[0], expected, rtol=1.0e-12).all())

    # Crank-Nicolson
    computed = ode_1st_linear_odeint(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1, rho=1.0)
    expected = ode_1st_linear_crankn(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1)
    assert(numpy.isclose(computed[0], expected, rtol=1.0e-12).all())

    # Generalised Alpha, rho = 0.5
    computed = ode_1st_linear_odeint(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1, rho=0.5)
    expected = ode_1st_linear_galpha(a=1.0, b=0.5, u_0=1.0, nT=10, dt=0.1, rho=0.5)
    assert(numpy.isclose(computed[0], expected, rtol=1.0e-12).all())

    # *** Check wrt convergence order

    # Parameter/order sets
    mi = {'euler_backward': {'param': {'alpha_f': 1.0, 'alpha_m': 0.5, 'gamma': 0.5}, 'order_expected': 1},
          'crank_nicolson': {'param': {'alpha_f': 0.5, 'alpha_m': 0.5, 'gamma': 0.5}, 'order_expected': 2},
          'generalised_alpha_rho_1.0': {'param': {'rho': 1.0}, 'order_expected': 2},
          'generalised_alpha_rho_0.5': {'param': {'rho': 0.5}, 'order_expected': 2},
          'generalised_alpha_rho_0.0': {'param': {'rho': 0.0}, 'order_expected': 2}, }

    # Compute error for each method and resolution
    for method, info in mi.items():

        dolfiny.utils.pprint(f"\n=== Processing method = {method}")

        l2 = {'u': {}, 'v': {}}

        for N in (50, 100, 200, 400):
            computed = ode_1st_linear_odeint(a=1.0, b=0.5, u_0=1.0, nT=N, dt=1.0 / N, **info['param'])
            expected = ode_1st_linear_closed(a=1.0, b=0.5, u_0=1.0, nT=N, dt=1.0 / N)
            u_c, v_c = computed
            u_e, v_e = expected
            #
            l2['u'][N] = numpy.linalg.norm(u_c - u_e, 2) / numpy.linalg.norm(u_e, 2)
            l2['v'][N] = numpy.linalg.norm(v_c - v_e, 2) / numpy.linalg.norm(v_e, 2)

        info["l2error"] = {}
        info["order_measured"] = {}

        for l2key, l2value in l2.items():

            # Get order of convergence from k finest studies
            k = 3
            x = numpy.log10(numpy.fromiter(l2value.keys(), dtype=float))
            y = numpy.log10(numpy.fromiter(l2value.values(), dtype=float))
            A = numpy.vstack([x[-k:], numpy.ones(k)]).T
            m = numpy.linalg.lstsq(A, y[-k:], rcond=None)[0][0]

            info["l2error"][l2key] = l2value
            info["order_measured"][l2key] = numpy.abs(m)

            assert(numpy.greater(info["order_measured"][l2key], info['order_expected'] - 0.05))

    # Output
    if MPI.COMM_WORLD.rank == 0:

        import json
        import matplotlib.pyplot
        import matplotlib.colors
        import itertools

        # Write results as json file
        with open('test_odeint_linear_convergence.json', 'w') as file:
            json.dump(mi, file, indent=3)

        # Plot results as pdf file
        with open('test_odeint_linear_convergence.json', 'r') as file:
            mi = json.load(file)
        fig, ax1 = matplotlib.pyplot.subplots()
        ax1.set_title("ODEInt: linear 1st order ODE, convergence", fontsize=12)
        ax1.set_xlabel(r'number of time steps $\log(N)$', fontsize=12)
        ax1.set_ylabel(r'L2 error $\log (e)$', fontsize=12)
        ax1.grid(linewidth=0.25)
        fig.tight_layout()
        markers = itertools.cycle(['o', 's', 'x', 'h', 'p', '+'])
        lstyles = itertools.cycle(['-', '--'])
        colours = itertools.cycle(matplotlib.colors.TABLEAU_COLORS)
        for method, info in mi.items():
            marker = next(markers)
            colour = next(colours)
            for l2key, l2value in info["l2error"].items():
                lstyle = next(lstyles)
                n = numpy.log10(numpy.fromiter(l2value.keys(), dtype=float))
                e = numpy.log10(numpy.fromiter(l2value.values(), dtype=float))
                label = method + " (" + l2key + ")" if l2key == 'u' else None
                ax1.plot(n, e, color=colour, linestyle=lstyle, marker=marker, linewidth=1, markersize=5, label=label)
        ax1.legend(loc='lower left')
        fig.savefig("test_odeint_linear_convergence.pdf")


def test_odeint_nonlinear():

    # *** Check wrt convergence order

    # Parameter/order sets
    mi = {'euler_backward': {'param': {'alpha_f': 1.0, 'alpha_m': 0.5, 'gamma': 0.5}, 'order_expected': 1},
          'crank_nicolson': {'param': {'alpha_f': 0.5, 'alpha_m': 0.5, 'gamma': 0.5}, 'order_expected': 2},
          'generalised_alpha_rho_1.0': {'param': {'rho': 1.0}, 'order_expected': 2},
          'generalised_alpha_rho_0.5': {'param': {'rho': 0.5}, 'order_expected': 2},
          'generalised_alpha_rho_0.0': {'param': {'rho': 0.0}, 'order_expected': 2}, }

    # Compute error for each method and resolution
    for method, info in mi.items():

        dolfiny.utils.pprint(f"\n=== Processing method = {method}")

        l2 = {'u': {}, 'v': {}}

        for N in (200, 400, 800, 1600):
            computed = ode_1st_nonlinear_odeint(a=5.0, b=1.0, c=8.0, nT=N, dt=1.0 / N, **info['param'])
            expected = ode_1st_nonlinear_closed(a=5.0, b=1.0, c=8.0, nT=N, dt=1.0 / N)
            u_c, v_c = computed
            u_e, v_e = expected
            #
            l2['u'][N] = numpy.linalg.norm(u_c - u_e, 2) / numpy.linalg.norm(u_e, 2)
            l2['v'][N] = numpy.linalg.norm(v_c - v_e, 2) / numpy.linalg.norm(v_e, 2)

        info["l2error"] = {}
        info["order_measured"] = {}

        for l2key, l2value in l2.items():

            # Get order of convergence from k finest studies
            k = 3
            x = numpy.log10(numpy.fromiter(l2value.keys(), dtype=float))
            y = numpy.log10(numpy.fromiter(l2value.values(), dtype=float))
            A = numpy.vstack([x[-k:], numpy.ones(k)]).T
            m = numpy.linalg.lstsq(A, y[-k:], rcond=None)[0][0]

            info["l2error"][l2key] = l2value
            info["order_measured"][l2key] = numpy.abs(m)

            assert(numpy.greater(info["order_measured"][l2key], info['order_expected'] - 0.05))

    # Output
    if MPI.COMM_WORLD.rank == 0:

        import json
        import matplotlib.pyplot
        import matplotlib.colors
        import itertools

        # Write results as json file
        with open('test_odeint_nonlinear_convergence.json', 'w') as file:
            json.dump(mi, file, indent=3)

        # Plot results as pdf file
        with open('test_odeint_nonlinear_convergence.json', 'r') as file:
            mi = json.load(file)
        fig, ax1 = matplotlib.pyplot.subplots()
        ax1.set_title("ODEInt: nonlinear 1st order ODE, convergence", fontsize=12)
        ax1.set_xlabel(r'number of time steps $\log(N)$', fontsize=12)
        ax1.set_ylabel(r'L2 error $\log (e)$', fontsize=12)
        ax1.grid(linewidth=0.25)
        fig.tight_layout()
        markers = itertools.cycle(['o', 's', 'x', 'h', 'p', '+'])
        lstyles = itertools.cycle(['-', '--'])
        colours = itertools.cycle(matplotlib.colors.TABLEAU_COLORS)
        for method, info in mi.items():
            marker = next(markers)
            colour = next(colours)
            for l2key, l2value in info["l2error"].items():
                lstyle = next(lstyles)
                n = numpy.log10(numpy.fromiter(l2value.keys(), dtype=float))
                e = numpy.log10(numpy.fromiter(l2value.values(), dtype=float))
                label = method + " (" + l2key + ")" if l2key == 'u' else None
                ax1.plot(n, e, color=colour, linestyle=lstyle, marker=marker, linewidth=1, markersize=5, label=label)
        ax1.legend(loc='lower left')
        fig.savefig("test_odeint_nonlinear_convergence.pdf")


def ode_2nd_linear_closed(a=0.1, b=1.0, c=1.0, u_0=1.0, du_0=1.0, nT=10, dt=0.1):
    """
    Solve ODE in closed form (analytically, at discrete time instances).

    Second order linear ODE:
    ddot u + a * dot u + b * u - c = 0 with initial conditions u(t=0) = u_0 ; du(t=0) = du_0
    """

    t = numpy.linspace(0, nT * dt, num=nT + 1)

    La = numpy.sqrt(4 * b - a**2)
    C2 = u_0 - c / b
    C1 = 2 / La * (du_0 + C2 * a / 2)

    u = numpy.exp(-0.5 * a * t) * (C1 * numpy.sin(La / 2 * t) + C2 * numpy.cos(La / 2 * t)) + c / b
    ut = numpy.exp(-a * t / 2) * ((C1 * La * numpy.cos(La * t / 2)) / 2 - (C2 * La * numpy.sin(La * t / 2)) / 2) \
        - (a * numpy.exp(-a * t / 2) * (C2 * numpy.cos(La * t / 2) + C1 * numpy.sin(La * t / 2))) / 2
    utt = (a**2 * numpy.exp(-a * t / 2) * (C2 * numpy.cos(La * t / 2) + C1 * numpy.sin(La * t / 2))) / 4 \
        - numpy.exp(-a * t / 2) * (C2 * La**2 * numpy.cos(La * t / 2) / 4 + (C1 * La**2 * numpy.sin(La * t / 2)) / 4) \
        - a * numpy.exp(-a * t / 2) * ((C1 * La * numpy.cos(La * t / 2)) / 2 - (C2 * La * numpy.sin(La * t / 2)) / 2)

    return u, ut, utt


def ode_2nd_linear_odeint(a=0.1, b=1.0, c=1.0, u_0=1.0, du_0=1.0, nT=10, dt=0.1, **kwargs):
    """
    Create 2nd order ODE problem and solve with `ODEInt` time integrator.

    Second order linear ODE:
    ddot u + a * dot u + b * u - c = 0 with initial conditions u(t=0) = u_0 ; du(t=0) = du_0
    """

    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 10)
    U = FunctionSpace(mesh, ("DG", 0))

    u = Function(U, name="u")
    ut = Function(U, name="ut")
    utt = Function(U, name="utt")

    u.vector.set(u_0)  # initial condition
    ut.vector.set(du_0)  # initial condition
    utt.vector.set(c - a * du_0 - b * u_0)  # exact initial rate of rate of this ODE for generalised alpha

    u.vector.ghostUpdate()
    ut.vector.ghostUpdate()
    utt.vector.ghostUpdate()

    δu = ufl.TestFunction(U)

    dx = ufl.Measure("dx", domain=mesh)

    # Global time
    time = dolfinx.Constant(mesh, 0.0)

    # Time step size
    dt = dolfinx.Constant(mesh, dt)

    # Time integrator
    odeint = dolfiny.odeint.ODEInt2(t=time, dt=dt, x=u, xt=ut, xtt=utt, **kwargs)

    # Strong form residual (as one-form)
    r = utt + a * ut + b * u - c

    # Weighted residual (as one-form)
    f = δu * r * dx

    # Overall form (as one-form)
    F = odeint.discretise_in_time(f)
    # Overall form (as list of forms)
    F = dolfiny.function.extract_blocks(F, [δu])

    # Silence SNES monitoring during test
    dolfiny.snesblockproblem.SNESBlockProblem.print_norms = lambda self, it: 1

    # Create problem (although having a linear ODE we use the dolfiny.snesblockproblem API)
    problem = dolfiny.snesblockproblem.SNESBlockProblem(F, [u])

    # Book-keeping of results
    u_, ut_, utt_ = numpy.zeros(nT + 1), numpy.zeros(nT + 1), numpy.zeros(nT + 1)
    u_[0], ut_[0], utt_[0] = [v.vector.sum() / v.vector.getSize() for v in [u, ut, utt]]

    dolfiny.utils.pprint(f"+++ Processing time steps = {nT}")

    # Process time steps
    for time_step in range(1, nT + 1):

        # Stage next time step
        odeint.stage()

        # Solve (linear) problem
        problem.solve()

        # Update solution states for time integration
        odeint.update()

        # Assert zero residual at t + dt
        assert numpy.isclose(dolfiny.expression.assemble(r, dx), 0.0, atol=1e-6), "Non-zero residual at (t + dt)!"

        # Store results
        u_[time_step], ut_[time_step], utt_[time_step] = [v.vector.sum() / v.vector.getSize() for v in [u, ut, utt]]

    return u_, ut_, utt_


def ode_2nd_nonlinear_closed(a=1.0, b=1.0, u_0=1.0, nT=10, dt=0.1):
    """
    Solve ODE in closed form (analytically, at discrete time instances).

    Second order nonlinear ODE: (Duffing)
    ddot u + a * u + b * u^3 = 0 with initial conditions u(t=0) = u_0 ; du(t=0) = 0
    """

    t = numpy.linspace(0, nT * dt, num=nT + 1)

    # Analytical solution in terms of Jacobi elliptic functions (exact)
    # u(t) = u_0 * cn(c * t, m)
    # ut(t) = -c * u_0 * dn(c * t, m) * sn(c * t, m)
    # utt(t) = -c^2 * u_0 * cn(c * t, m) * dn(c * t, m) * (dn(c * t, m) - m * sd(c * t, m) * sn(c * t, m))
    import scipy.special
    c = numpy.sqrt(a + b * u_0**2)
    k = b * u_0**2 / 2 / (a + b * u_0**2)
    # Capture cases for which the modulus k is not within the [0,1] interval of `scipy.special.ellipj`.
    # This is needed for a softening Duffing oscillator with b < 0.
    # Arguments for `scipy.special.ellipj`
    u = c * t
    m = k
    #
    if m >= 0 and m <= 1:
        sn, cn, dn, _ = scipy.special.ellipj(u, m)
        sn_, cn_, dn_ = sn, cn, dn
    if m > 1:
        u_ = u * m**(1 / 2)
        m_ = m**(-1)
        sn, cn, dn, _ = scipy.special.ellipj(u_, m_)
        sn_, cn_, dn_ = m**(-1 / 2) * sn, dn, cn
    if m < 0:
        u_ = u * (1 / (1 - m))**(-1 / 2)
        m_ = -m / (1 - m)
        sn, cn, dn, _ = scipy.special.ellipj(u_, m_)
        sn_, cn_, dn_ = (1 / (1 - m))**(1 / 2) * sn / dn, cn / dn, 1 / dn

    u = u_0 * cn_
    ut = -c * u_0 * dn_ * sn_
    utt = -c**2 * u_0 * cn_ * dn_ * (dn_ - m * sn_ / dn_ * sn_)

    return u, ut, utt


def ode_2nd_nonlinear_odeint(a=1.0, b=1.0, u_0=1.0, nT=10, dt=0.1, **kwargs):
    """
    Create 2nd order ODE problem and solve with `ODEInt` time integrator.

    Second order nonlinear ODE: (Duffing)
    ddot u + a * u + b * u^3 = 0 with initial conditions u(t=0) = u_0 ; du(t=0) = 0
    """

    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 10)
    U = FunctionSpace(mesh, ("DG", 0))

    u = Function(U, name="u")
    ut = Function(U, name="ut")
    utt = Function(U, name="utt")

    u.vector.set(u_0)  # initial condition
    ut.vector.set(0.0)  # initial condition
    utt.vector.set(- a * u_0 - b * u_0**3)  # exact initial rate of rate of this ODE for generalised alpha

    u.vector.ghostUpdate()
    ut.vector.ghostUpdate()
    utt.vector.ghostUpdate()

    δu = ufl.TestFunction(U)

    dx = ufl.Measure("dx", domain=mesh)

    # Global time
    time = dolfinx.Constant(mesh, 0.0)

    # Time step size
    dt = dolfinx.Constant(mesh, dt)

    # Time integrator
    odeint = dolfiny.odeint.ODEInt2(t=time, dt=dt, x=u, xt=ut, xtt=utt, **kwargs)

    # Strong form residual (as one-form)
    r = utt + a * u + b * u**3

    # Weighted residual (as one-form)
    f = δu * r * dx

    # Overall form (as one-form)
    F = odeint.discretise_in_time(f)
    # Overall form (as list of forms)
    F = dolfiny.function.extract_blocks(F, [δu])

    # Options for PETSc backend
    from petsc4py import PETSc
    opts = PETSc.Options()
    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_atol"] = 1.0e-09
    opts["snes_rtol"] = 1.0e-12

    # Silence SNES monitoring during test
    dolfiny.snesblockproblem.SNESBlockProblem.print_norms = lambda self, it: 1

    # Create nonlinear problem
    problem = dolfiny.snesblockproblem.SNESBlockProblem(F, [u])

    # Book-keeping of results
    u_, ut_, utt_ = numpy.zeros(nT + 1), numpy.zeros(nT + 1), numpy.zeros(nT + 1)
    u_[0], ut_[0], utt_[0] = [v.vector.sum() / v.vector.getSize() for v in [u, ut, utt]]

    dolfiny.utils.pprint(f"+++ Processing time steps = {nT}")

    # Process time steps
    for time_step in range(1, nT + 1):

        # Stage next time step
        odeint.stage()

        # Solve nonlinear problem
        problem.solve()

        # Assert convergence of nonlinear solver
        assert problem.snes.getConvergedReason() > 0, "Nonlinear solver did not converge!"

        # Update solution states for time integration
        odeint.update()

        # Assert zero residual at t + dt
        assert numpy.isclose(dolfiny.expression.assemble(r, dx), 0.0, atol=1e-6), "Non-zero residual at (t + dt)!"

        # Store results
        u_[time_step], ut_[time_step], utt_[time_step] = [v.vector.sum() / v.vector.getSize() for v in [u, ut, utt]]

    return u_, ut_, utt_


def test_odeint2_linear():

    # *** Check wrt convergence order

    # Parameter/order sets
    mi = {'newmark':
          {'param': {'alpha_f': 1.0, 'alpha_m': 1.0, 'gamma': 0.5, 'beta': 0.25}, 'order_expected': 2},
          'hht_alpha_rho_0.5':
          {'param': {'alpha_f': 2 / 3, 'alpha_m': 1.0, 'gamma': 5 / 6, 'beta': 4 / 9}, 'order_expected': 2},
          'wbz_alpha_rho_0.5':
          {'param': {'alpha_f': 1.0, 'alpha_m': 4 / 3, 'gamma': 5 / 6, 'beta': 4 / 9}, 'order_expected': 2},
          'generalised_alpha_rho_1.0': {'param': {'rho': 1.0}, 'order_expected': 2},
          'generalised_alpha_rho_0.5': {'param': {'rho': 0.5}, 'order_expected': 2},
          'generalised_alpha_rho_0.0': {'param': {'rho': 0.0}, 'order_expected': 2}, }

    # Compute error for each method and resolution
    for method, info in mi.items():

        dolfiny.utils.pprint(f"\n=== Processing method = {method}")

        l2 = {'u': {}, 'v': {}, 'a': {}}

        for N in (50, 100, 200, 400):
            computed = ode_2nd_linear_odeint(a=12, b=1e3, c=1e3, u_0=0.5, du_0=0.0, nT=N, dt=1.0 / N, **info['param'])
            expected = ode_2nd_linear_closed(a=12, b=1e3, c=1e3, u_0=0.5, du_0=0.0, nT=N, dt=1.0 / N)            #
            u_c, v_c, a_c = computed
            u_e, v_e, a_e = expected
            #
            l2['u'][N] = numpy.linalg.norm(u_c - u_e, 2) / numpy.linalg.norm(u_e, 2)
            l2['v'][N] = numpy.linalg.norm(v_c - v_e, 2) / numpy.linalg.norm(v_e, 2)
            l2['a'][N] = numpy.linalg.norm(a_c - a_e, 2) / numpy.linalg.norm(a_e, 2)

        info["l2error"] = {}
        info["order_measured"] = {}

        for l2key, l2value in l2.items():

            # Get order of convergence from k finest studies
            k = 3
            x = numpy.log10(numpy.fromiter(l2value.keys(), dtype=float))
            y = numpy.log10(numpy.fromiter(l2value.values(), dtype=float))
            A = numpy.vstack([x[-k:], numpy.ones(k)]).T
            m = numpy.linalg.lstsq(A, y[-k:], rcond=None)[0][0]

            info["l2error"][l2key] = l2value
            info["order_measured"][l2key] = numpy.abs(m)

            assert(numpy.greater(info["order_measured"][l2key], info['order_expected'] - 0.15))

    # Output
    if MPI.COMM_WORLD.rank == 0:

        import json
        import matplotlib.pyplot
        import matplotlib.colors
        import itertools

        # Write results as json file
        with open('test_odeint2_linear_convergence.json', 'w') as file:
            json.dump(mi, file, indent=3)

        # Plot results as pdf file
        with open('test_odeint2_linear_convergence.json', 'r') as file:
            mi = json.load(file)
        fig, ax1 = matplotlib.pyplot.subplots()
        ax1.set_title("ODEInt2: linear 2nd order ODE, convergence", fontsize=12)
        ax1.set_xlabel(r'number of time steps $\log(N)$', fontsize=12)
        ax1.set_ylabel(r'L2 error $\log (e)$', fontsize=12)
        ax1.grid(linewidth=0.25)
        fig.tight_layout()
        markers = itertools.cycle(['o', 's', 'x', 'h', 'p', '+'])
        lstyles = itertools.cycle(['-', '--', ':'])
        colours = itertools.cycle(matplotlib.colors.TABLEAU_COLORS)
        for method, info in mi.items():
            marker = next(markers)
            colour = next(colours)
            for l2key, l2value in info["l2error"].items():
                lstyle = next(lstyles)
                n = numpy.log10(numpy.fromiter(l2value.keys(), dtype=float))
                e = numpy.log10(numpy.fromiter(l2value.values(), dtype=float))
                label = method + " (" + l2key + ")" if l2key == 'u' else None
                ax1.plot(n, e, color=colour, linestyle=lstyle, marker=marker, linewidth=1, markersize=5, label=label)
        ax1.legend(loc='lower left')
        fig.savefig("test_odeint2_linear_convergence.pdf")


def test_odeint2_nonlinear():

    # *** Check wrt convergence order

    # Parameter/order sets
    mi = {'newmark':
          {'param': {'alpha_f': 1.0, 'alpha_m': 1.0, 'gamma': 0.5, 'beta': 0.25}, 'order_expected': 2},
          'hht_alpha_rho_0.5':
          {'param': {'alpha_f': 2 / 3, 'alpha_m': 1.0, 'gamma': 5 / 6, 'beta': 4 / 9}, 'order_expected': 2},
          'wbz_alpha_rho_0.5':
          {'param': {'alpha_f': 1.0, 'alpha_m': 4 / 3, 'gamma': 5 / 6, 'beta': 4 / 9}, 'order_expected': 2},
          'generalised_alpha_rho_1.0': {'param': {'rho': 1.0}, 'order_expected': 2},
          'generalised_alpha_rho_0.5': {'param': {'rho': 0.5}, 'order_expected': 2},
          'generalised_alpha_rho_0.0': {'param': {'rho': 0.0}, 'order_expected': 2}, }

    # Compute error for each method and resolution
    for method, info in mi.items():

        dolfiny.utils.pprint(f"\n=== Processing method = {method}")

        l2 = {'u': {}, 'v': {}, 'a': {}}

        for N in (200, 400, 800, 1600):
            computed = ode_2nd_nonlinear_odeint(a=3000.0, b=-2500.0, u_0=1.0, nT=N, dt=1.0 / N, **info['param'])
            expected = ode_2nd_nonlinear_closed(a=3000.0, b=-2500.0, u_0=1.0, nT=N, dt=1.0 / N)
            #
            u_c, v_c, a_c = computed
            u_e, v_e, a_e = expected
            #
            l2['u'][N] = numpy.linalg.norm(u_c - u_e, 2) / numpy.linalg.norm(u_e, 2)
            l2['v'][N] = numpy.linalg.norm(v_c - v_e, 2) / numpy.linalg.norm(v_e, 2)
            l2['a'][N] = numpy.linalg.norm(a_c - a_e, 2) / numpy.linalg.norm(a_e, 2)

        info["l2error"] = {}
        info["order_measured"] = {}

        for l2key, l2value in l2.items():

            # Get order of convergence from k finest studies
            k = 3
            x = numpy.log10(numpy.fromiter(l2value.keys(), dtype=float))
            y = numpy.log10(numpy.fromiter(l2value.values(), dtype=float))
            A = numpy.vstack([x[-k:], numpy.ones(k)]).T
            m = numpy.linalg.lstsq(A, y[-k:], rcond=None)[0][0]

            info["l2error"][l2key] = l2value
            info["order_measured"][l2key] = numpy.abs(m)

            assert(numpy.greater(info["order_measured"][l2key], info['order_expected'] - 0.05))

    # Output
    if MPI.COMM_WORLD.rank == 0:

        import json
        import matplotlib.pyplot
        import matplotlib.colors
        import itertools

        # Write results as json file
        with open('test_odeint2_nonlinear_convergence.json', 'w') as file:
            json.dump(mi, file, indent=3)

        # Plot results as pdf file
        with open('test_odeint2_nonlinear_convergence.json', 'r') as file:
            mi = json.load(file)
        fig, ax1 = matplotlib.pyplot.subplots()
        ax1.set_title("ODEInt2: nonlinear 2nd order ODE, convergence", fontsize=12)
        ax1.set_xlabel(r'number of time steps $\log(N)$', fontsize=12)
        ax1.set_ylabel(r'L2 error $\log (e)$', fontsize=12)
        ax1.grid(linewidth=0.25)
        fig.tight_layout()
        markers = itertools.cycle(['o', 's', 'x', 'h', 'p', '+'])
        lstyles = itertools.cycle(['-', '--', ':'])
        colours = itertools.cycle(matplotlib.colors.TABLEAU_COLORS)
        for method, info in mi.items():
            marker = next(markers)
            colour = next(colours)
            for l2key, l2value in info["l2error"].items():
                lstyle = next(lstyles)
                n = numpy.log10(numpy.fromiter(l2value.keys(), dtype=float))
                e = numpy.log10(numpy.fromiter(l2value.values(), dtype=float))
                label = method + " (" + l2key + ")" if l2key == 'u' else None
                ax1.plot(n, e, color=colour, linestyle=lstyle, marker=marker, linewidth=1, markersize=5, label=label)
        ax1.legend(loc='lower left')
        fig.savefig("test_odeint2_nonlinear_convergence.pdf")
