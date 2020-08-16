import dolfinx
import dolfiny.expression
import dolfiny.interpolation


class ODEInt():

    def __init__(self, t, dt, x, xt, **kwargs):
        """Initialises the ODE integrator (single-step-method).
        Uses underneath the generalised alpha method and its limits:

        Euler forward:
            alpha_f = 0, alpha_m = 1/2, gamma = 1/2
        Euler backward:
            alpha_f = 1, alpha_m = 1/2, gamma = 1/2
        Crank-Nicolson:
            alpha_f = 1/2, alpha_m = 1/2, gamma = 1/2
        Theta:
            alpha_f = theta, alpha_m = 1/2, gamma = 1/2
        Generalised alpha:
            The value of rho can be used to determine the values
            alpha_f = 1 / (1 + rho),
            alpha_m = 1 / 2 * (3 - rho) / (1 + rho),
            gamma = 1 / 2 + alpha_m - alpha_f

        Parameters
        ----------
        t: Stage time.
        dt: Time step size.
        x: Pointer to function describing the state.
        xt: Pointer to function describing the rate of the state.

        rho: Spectral radius rho_infinity for generalised alpha.
        alpha_f: Specific value for alpha_f.
        alpha_m: Specific value for alpha_m.
        gamma: Specific value for gamma.
        """

        # Set stage time and time step
        self.t = t
        self.dt = dt

        if not isinstance(self.t, dolfinx.Constant):
            raise RuntimeError("No stage time t as dolfinx.Constant provied.")

        if not isinstance(self.dt, dolfinx.Constant):
            raise RuntimeError("No time step dt as dolfinx.Constant provied.")

        # Pointers to state x and rate xt (as function or list of functions)
        self.x = x
        self.xt = xt

        if isinstance(self.x, list):
            self.x0 = []
            for x, xt in zip(self.x, self.xt):
                if x.function_space is not xt.function_space:
                    raise RuntimeError("Incompatible function spaces for state and rate.")
        else:
            if self.x.function_space is not self.xt.function_space:
                raise RuntimeError("Incompatible function spaces for state and rate.")

        # Set state x0
        if isinstance(self.x, list):
            self.x0 = []
            for xi in self.x:
                self.x0.append(dolfinx.function.Function(xi.function_space))
        else:
            self.x0 = dolfinx.function.Function(self.x.function_space)

        # Set rate of state x0t
        if isinstance(self.xt, list):
            self.x0t = []
            for xti in self.xt:
                self.x0t.append(dolfinx.function.Function(xti.function_space))
        else:
            self.x0t = dolfinx.function.Function(self.xt.function_space)

        # Expression: derivative in time
        self.derivative_dt = lambda x, x0, x0t: \
            1.0 / (self.gamma * self.dt) * (x - x0) + (self.gamma - 1.0) / self.gamma * x0t

        # Expression: integral in time
        self.integral_dt = lambda x, xt, x0, x0t: \
            self.dt / 2 * (x0 + x) + self.dt**2 / 12 * (x0t - xt)

        # Expression: state at collocation point in time interval
        self.state = lambda x0, x1: \
            self.alpha_f * x1 + (1.0 - self.alpha_f) * x0

        # Expression: rate of state at collocation point in time interval
        self.rate = lambda x0t, x1t: \
            self.alpha_m * x1t + (1.0 - self.alpha_m) * x0t

        # Default values: Backward Euler
        self.alpha_f = dolfinx.Constant(self.t.ufl_domain(), 1.0)
        self.alpha_m = dolfinx.Constant(self.t.ufl_domain(), 0.5)
        self.gamma = dolfinx.Constant(self.t.ufl_domain(), 0.5)

        # Parameters from given rho
        if "rho" in kwargs:
            self.alpha_f.value = 1.0 / (1.0 + kwargs["rho"])
            self.alpha_m.value = 0.5 * (3.0 - kwargs["rho"]) * self.alpha_f.value
            self.gamma.value = 0.5 + self.alpha_m.value - self.alpha_f.value

        # Parameters directly
        if "alpha_f" in kwargs and "alpha_m" in kwargs and "gamma" in kwargs:
            self.alpha_f.value = kwargs["alpha_f"]
            self.alpha_m.value = kwargs["alpha_m"]
            self.gamma.value = kwargs["gamma"]

    def stage(self, t0=None, dt=None):

        if t0 is not None:
            self.t.value = t0

        if dt is not None:
            self.dt.value = dt

        self.t.value += self.alpha_f.value * self.dt.value

        # update x0 (copy values)
        if isinstance(self.x, list):
            for xi, x0i in zip(self.x, self.x0):
                with xi.vector.localForm() as locxi, x0i.vector.localForm() as locx0i:
                    locxi.copy(locx0i)
        else:
            with self.x.vector.localForm() as locx, self.x0.vector.localForm() as locx0:
                locx.copy(locx0)

        # update x0t (copy values)
        if isinstance(self.xt, list):
            for xti, x0ti in zip(self.xt, self.x0t):
                with xti.vector.localForm() as locxti, x0ti.vector.localForm() as locx0ti:
                    locxti.copy(locx0ti)
        else:
            with self.xt.vector.localForm() as locxt, self.x0t.vector.localForm() as locx0t:
                locxt.copy(locx0t)

        return self.t, self.dt

    def update(self):

        # update xt
        if isinstance(self.xt, list):
            for x, xt, x0, x0t in zip(self.x, self.xt, self.x0, self.x0t):
                dolfiny.interpolation.interpolate(self.derivative_dt(x, x0, x0t), xt)
        else:
            dolfiny.interpolation.interpolate(self.derivative_dt(self.x, self.x0, self.x0t), self.xt)

        # update to final time of staged time step
        self.t.value += (1.0 - self.alpha_f.value) * self.dt.value

    def discretise_in_time(self, f):

        # xa
        if isinstance(self.x, list):
            xa = []
            for x0, x1 in zip(self.x0, self.x):
                xa.append(self.state(x0, x1))
        else:
            xa = self.state(self.x0, self.x)

        # xat
        if isinstance(self.x, list):
            xat = []
            for x, x0, x0t in zip(self.x, self.x0, self.x0t):
                x1t = self.derivative_dt(x, x0, x0t)
                xat.append(self.rate(x0t, x1t))
        else:
            x1t = self.derivative_dt(self.x, self.x0, self.x0t)
            xat = self.rate(self.x0t, x1t)

        f = dolfiny.expression.evaluate(f, self.x, xa)
        f = dolfiny.expression.evaluate(f, self.xt, xat)

        return f
