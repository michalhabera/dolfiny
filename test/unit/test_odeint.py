import pytest
import numpy

import standalone_odeint as m


@pytest.mark.parametrize("odeint_m, closed_m", [
    (m.ode_1st_linear_odeint, m.ode_1st_linear_closed),
    (m.ode_1st_nonlinear_odeint, m.ode_1st_nonlinear_closed),
    (m.ode_2nd_linear_odeint, m.ode_2nd_linear_closed),
    (m.ode_2nd_nonlinear_odeint, m.ode_2nd_nonlinear_closed),
    (m.ode_1st_nonlinear_mdof_odeint, m.ode_2nd_nonlinear_closed),
])
def test_odeint_accuracy(odeint_m, closed_m):

    computed = odeint_m()
    expected = closed_m()

    assert numpy.isclose(computed[0], expected[0], rtol=0.01, atol=0.01).all()
