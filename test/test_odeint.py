from mpi4py import MPI

import dolfinx
from dolfinx.generation import UnitIntervalMesh
from dolfinx.function import Function, FunctionSpace

import ufl
import numpy
import json

import dolfiny.odeint
import dolfiny.function
import dolfiny.snesblockproblem

import pytest


@pytest.fixture
def generalised_alpha_1st_params():
    return {
        'euler_backward': {'param': {'alpha_f': 1.0, 'alpha_m': 0.5, 'gamma': 0.5}, 'order_expected': 1},
        'crank_nicolson': {'param': {'alpha_f': 0.5, 'alpha_m': 0.5, 'gamma': 0.5}, 'order_expected': 2},
        'generalised_alpha_rho_1.0': {'param': {'rho': 1.0}, 'order_expected': 2},
        'generalised_alpha_rho_0.5': {'param': {'rho': 0.5}, 'order_expected': 2},
        'generalised_alpha_rho_0.0': {'param': {'rho': 0.0}, 'order_expected': 2},
    }


@pytest.fixture
def generalised_alpha_2nd_params():
    return {
        'newmark': {'param': {'alpha_f': 1.0, 'alpha_m': 1.0, 'gamma': 0.5, 'beta': 0.25}, 'order_expected': 2},
        'hht_rho_0.5': {'param': {'alpha_f': 2/3, 'alpha_m': 1.0, 'gamma': 5/6, 'beta': 4/9}, 'order_expected': 2},  # noqa: E226, E501
        'wbz_rho_0.5': {'param': {'alpha_f': 1.0, 'alpha_m': 4/3, 'gamma': 5/6, 'beta': 4/9}, 'order_expected': 2},  # noqa: E226, E501
        'generalised_alpha_rho_1.0': {'param': {'rho': 1.0}, 'order_expected': 2},
        'generalised_alpha_rho_0.5': {'param': {'rho': 0.5}, 'order_expected': 2},
        'generalised_alpha_rho_0.0': {'param': {'rho': 0.0}, 'order_expected': 2},
    }


# === ODEInt-based solutions =================================================


def ode_1st_linear_odeint(a=1.0, b=0.5, u_0=1.0, nT=100, dt=0.01, **kwargs):
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


def ode_1st_nonlinear_odeint(a=5.0, b=1.0, c=8.0, nT=400, dt=0.0025, **kwargs):
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


def ode_2nd_linear_odeint(a=12.0, b=1000.0, c=1000.0, u_0=0.5, du_0=0.0, nT=100, dt=0.01, **kwargs):
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


def ode_2nd_nonlinear_odeint(a=100, b=-50, u_0=1.0, nT=400, dt=0.0025, **kwargs):
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


def ode_1st_nonlinear_mdof_odeint(a=100, b=-50, u_0=1.0, nT=400, dt=0.0025, **kwargs):
    """
    First order nonlinear system of ODEs: (Duffing oscillator, undamped, unforced)

    (1) dot v + s = 0
    (2) dot s - st(u, v) = 0

    (*) dot u - v = 0

    with the constitutive law: s(u) = [a + b * u^2] * u
    and its rate form: st(u, v) = [a + 3 * b * u^2] * v
    and initial conditions: u(t=0) = u_0, v(t=0) = v_0 and s(t=0) = s(u_0)
    """

    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 10)

    # Problem parameters, note: (a + b * u_0**2) !> 0
    a, b = a, b

    # Initial conditions
    u_0, v_0 = u_0, 0.0

    def _s(u):
        return (a + b * u**2) * u  # constitutive law

    def _st(u, v):
        return (a + 3 * b * u**2) * v  # rate of constitutive law

    V = FunctionSpace(mesh, ("DG", 0))
    S = FunctionSpace(mesh, ("DG", 0))

    v = Function(V, name="v")
    s = Function(S, name="s")
    vt = Function(V, name="vt")
    st = Function(S, name="st")

    u = Function(V, name="u")
    d = Function(V, name="d")  # dummy

    δv = ufl.TestFunction(V)
    δs = ufl.TestFunction(S)

    m, mt, δm = [v, s], [vt, st], [δv, δs]

    # Set initial conditions
    v.vector.set(v_0), vt.vector.set(-_s(u_0))
    s.vector.set(_s(u_0)), st.vector.set(_st(u_0, v_0))
    u.vector.set(u_0)

    [w.vector.ghostUpdate() for w in [v, s, u, vt, st]]

    # Measure
    dx = ufl.Measure("dx", domain=mesh)

    # Number of time steps
    nT = nT

    # Global time
    t = dolfinx.Constant(mesh, 0.0)

    # Time step size
    dt = dolfinx.Constant(mesh, dt)

    # Time integrator
    odeint = dolfiny.odeint.ODEInt(t=t, dt=dt, x=m, xt=mt, **kwargs)

    # Expression for time-integrated quantities
    u_expr = u + odeint.integral_dt(v)

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
    from petsc4py import PETSc
    opts = PETSc.Options()
    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_atol"] = 1.0e-09
    opts["snes_rtol"] = 1.0e-12

    # Silence SNES monitoring during test
    dolfiny.snesblockproblem.SNESBlockProblem.print_norms = lambda self, it: 1

    # Create nonlinear problem
    problem = dolfiny.snesblockproblem.SNESBlockProblem(F, m)

    # Book-keeping of results
    u_, v_, vt_ = [numpy.zeros(nT + 1) for w in [u, v, vt]]
    u_[0], v_[0], vt_[0] = [w.vector.sum() / w.vector.getSize() for w in [u, v, vt]]

    dolfiny.utils.pprint(f"+++ Processing time steps = {nT}")

    # Process time steps
    for ts in range(1, nT + 1):

        # Stage next time step
        odeint.stage()

        # Solve nonlinear problem
        problem.solve()

        # Assert convergence of nonlinear solver
        assert problem.snes.getConvergedReason() > 0, "Nonlinear solver did not converge!"

        # Update solution states for time integration
        odeint.update()

        # Assert zero residual at t + dt
        assert numpy.isclose(dolfiny.expression.assemble(r1, dx), 0.0, atol=1e-6), "Non-zero residual r1 at (t + dt)!"
        assert numpy.isclose(dolfiny.expression.assemble(r2, dx), 0.0, atol=1e-6), "Non-zero residual r2 at (t + dt)!"

        # Assign time-integrated quantities
        dolfiny.interpolation.interpolate(u_expr, d)
        dolfiny.interpolation.interpolate(d, u)

        # Store results
        u_[ts], v_[ts], vt_[ts] = [w.vector.sum() / w.vector.getSize() for w in [u, v, vt]]

    return u_, v_, vt_


# === Closed-form solutions ==================================================

def ode_1st_linear_closed(a=1.0, b=0.5, u_0=1.0, nT=100, dt=0.01):
    """
    Solve ODE in closed form (analytically, at discrete time instances).

    First order linear ODE:
    dot u + a * u - b = 0 with initial condition u(t=0) = u_0
    """

    t = numpy.linspace(0, nT * dt, num=nT + 1)

    u = (u_0 - b / a) * numpy.exp(-a * t) + b / a
    ut = -a * (u_0 - b / a) * numpy.exp(-a * t)

    return u, ut


def ode_1st_nonlinear_closed(a=5.0, b=1.0, c=8.0, nT=400, dt=0.0025):
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


def ode_2nd_linear_closed(a=12.0, b=1000.0, c=1000.0, u_0=0.5, du_0=0.0, nT=100, dt=0.01):
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


def ode_2nd_nonlinear_closed(a=100, b=-50, u_0=1.0, nT=400, dt=0.0025):
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


# === Tests and Convergence tests ============================================


@pytest.mark.parametrize("odeint_m, closed_m", [
    (ode_1st_linear_odeint, ode_1st_linear_closed),
    (ode_1st_nonlinear_odeint, ode_1st_nonlinear_closed),
    (ode_2nd_linear_odeint, ode_2nd_linear_closed),
    (ode_2nd_nonlinear_odeint, ode_2nd_nonlinear_closed),
])
def test_odeint_accuracy(odeint_m, closed_m):

    computed = odeint_m()
    expected = closed_m()

    assert numpy.isclose(computed[0], expected[0], rtol=0.05).all()


@pytest.mark.convergence
@pytest.mark.parametrize("jsonfile, odeint_m, closed_m", [
    ("test_odeint_linear_convergence.json", ode_1st_linear_odeint, ode_1st_linear_closed),
    ("test_odeint_nonlinear_convergence.json", ode_1st_nonlinear_odeint, ode_1st_nonlinear_closed),
    ("test_odeint_nonlinear_mdof_convergence.json", ode_1st_nonlinear_mdof_odeint, ode_2nd_nonlinear_closed),
])
def test_odeint_convergence(generalised_alpha_1st_params, jsonfile, odeint_m, closed_m):

    # Compute error for each method and resolution
    for method, info in generalised_alpha_1st_params.items():

        dolfiny.utils.pprint(f"\n=== Processing method = {method}")

        l2 = {'u': {}, 'v': {}}

        for N in (200, 400, 800, 1600):
            computed = odeint_m(nT=N, dt=1.0 / N, **info['param'])
            expected = closed_m(nT=N, dt=1.0 / N)
            #
            u_c, v_c = computed[:2]
            u_e, v_e = expected[:2]
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

            assert(numpy.greater(info["order_measured"][l2key], info['order_expected'] - 0.1))

    # Write results as json file
    with open(jsonfile, 'w') as file:
        json.dump(generalised_alpha_1st_params, file, indent=3)


@pytest.mark.convergence
@pytest.mark.parametrize("jsonfile, odeint_m, closed_m", [
    ("test_odeint2_linear_convergence.json", ode_2nd_linear_odeint, ode_2nd_linear_closed),
    ("test_odeint2_nonlinear_convergence.json", ode_2nd_nonlinear_odeint, ode_2nd_nonlinear_closed),
])
def test_odeint2_convergence(generalised_alpha_2nd_params, jsonfile, odeint_m, closed_m):

    # Compute error for each method and resolution
    for method, info in generalised_alpha_2nd_params.items():

        dolfiny.utils.pprint(f"\n=== Processing method = {method}")

        l2 = {'u': {}, 'v': {}, 'a': {}}

        for N in (200, 400, 800, 1600):
            computed = odeint_m(nT=N, dt=1.0 / N, **info['param'])
            expected = closed_m(nT=N, dt=1.0 / N)
            #
            u_c, v_c, a_c = computed[:3]
            u_e, v_e, a_e = expected[:3]
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

            assert(numpy.greater(info["order_measured"][l2key], info['order_expected'] - 0.1))

    # Write results as json file
    with open(jsonfile, 'w') as file:
        json.dump(generalised_alpha_2nd_params, file, indent=3)


@pytest.mark.convergence
@pytest.mark.parametrize("jsonfile, title", [
    ("test_odeint_linear_convergence.json", "ODEInt: linear 1st order ODE, convergence"),
    ("test_odeint_nonlinear_convergence.json", "ODEInt: nonlinear 1st order ODE, convergence"),
    ("test_odeint_nonlinear_mdof_convergence.json", "ODEInt: 2x nonlinear 1st order ODEs, convergence"),
    ("test_odeint2_linear_convergence.json", "ODEInt2: linear 2nd order ODE, convergence"),
    ("test_odeint2_nonlinear_convergence.json", "ODEInt2: nonlinear 2nd order ODE, convergence"),
])
def test_odeint_convergence_plot(jsonfile, title):

    import matplotlib.pyplot
    import matplotlib.colors
    import itertools

    with open(jsonfile, 'r') as file:
        mi = json.load(file)

    fig, ax1 = matplotlib.pyplot.subplots()
    ax1.set_title(title, fontsize=12)
    ax1.set_xlabel(r'number of steps $\log(N)$', fontsize=12)
    ax1.set_ylabel(r'L2 error $\log (e)$', fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()
    markers = itertools.cycle(['o', 's', 'x', 'h', 'p', '+'])
    colours = itertools.cycle(matplotlib.colors.TABLEAU_COLORS)
    for method, info in mi.items():
        marker = next(markers)
        colour = next(colours)
        lstyles = itertools.cycle(['-', '--', ':'])
        for l2key, l2value in info["l2error"].items():
            lstyle = next(lstyles)
            n = numpy.log10(numpy.fromiter(l2value.keys(), dtype=float))
            e = numpy.log10(numpy.fromiter(l2value.values(), dtype=float))
            label = method + " (" + l2key + ")" if l2key == 'u' else None
            ax1.plot(n, e, color=colour, linestyle=lstyle, marker=marker, linewidth=1, markersize=5, label=label)
    ax1.legend(loc='lower left')
    fig.savefig(jsonfile + ".pdf")
