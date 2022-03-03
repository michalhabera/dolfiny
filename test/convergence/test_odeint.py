import json

import dolfiny
import numpy
import pytest
import standalone_odeint as m
import standalone_postprocess as p


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
        'hht_rho_0.5': {'param': {'alpha_f': 2 / 3, 'alpha_m': 1.0, 'gamma': 5 / 6, 'beta': 4 / 9},
                        'order_expected': 2},
        'wbz_rho_0.5': {'param': {'alpha_f': 1.0, 'alpha_m': 4 / 3, 'gamma': 5 / 6, 'beta': 4 / 9},
                        'order_expected': 2},
        'generalised_alpha_rho_1.0': {'param': {'rho': 1.0}, 'order_expected': 2},
        'generalised_alpha_rho_0.5': {'param': {'rho': 0.5}, 'order_expected': 2},
        'generalised_alpha_rho_0.0': {'param': {'rho': 0.0}, 'order_expected': 2},
    }


@pytest.mark.convergence
@pytest.mark.parametrize("jsonfile, odeint_m, closed_m", [
    ("test_odeint_linear_convergence.json", m.ode_1st_linear_odeint, m.ode_1st_linear_closed),
    ("test_odeint_nonlinear_convergence.json", m.ode_1st_nonlinear_odeint, m.ode_1st_nonlinear_closed),
    ("test_odeint_nonlinear_mdof_convergence.json", m.ode_1st_nonlinear_mdof_odeint, m.ode_2nd_nonlinear_closed),
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
    ("test_odeint2_linear_convergence.json", m.ode_2nd_linear_odeint, m.ode_2nd_linear_closed),
    ("test_odeint2_nonlinear_convergence.json", m.ode_2nd_nonlinear_odeint, m.ode_2nd_nonlinear_closed),
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


@pytest.mark.postprocess
@pytest.mark.parametrize("jsonfile, title", [
    ("test_odeint_linear_convergence.json", "ODEInt: linear 1st order ODE, convergence"),
    ("test_odeint_nonlinear_convergence.json", "ODEInt: nonlinear 1st order ODE, convergence"),
    ("test_odeint_nonlinear_mdof_convergence.json", "ODEInt: 2x nonlinear 1st order ODEs, convergence"),
])
def test_odeint_postprocess(jsonfile, title):

    p.plot_convergence(jsonfile, title)


@pytest.mark.postprocess
@pytest.mark.parametrize("jsonfile, title", [
    ("test_odeint2_linear_convergence.json", "ODEInt2: linear 2nd order ODE, convergence"),
    ("test_odeint2_nonlinear_convergence.json", "ODEInt2: nonlinear 2nd order ODE, convergence"),
])
def test_odeint2_postprocess(jsonfile, title):

    p.plot_convergence(jsonfile, title)
