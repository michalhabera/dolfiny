import dolfinx
import dolfiny.expression


class ODEInt():

    def __init__(self, t, dt, **kwargs):
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
        rho: Spectral radius rho_infinity for generalised alpha.
        alpha_f: Specific value for alpha_f.
        alpha_m: Specific value for alpha_m.
        gamma: Specific value for gamma.
        x: Pointer to function describing the state at the end of time step -> x(t_end).
        x0: Pointer to function describing the state at the begin of time step -> x(t_begin).
        x0t: Pointer to function describing the rate at the begin of time step -> dx(t_begin)/dt.
        """
        # Set stage time and time step
        self.t = t
        self.dt = dt

        if not isinstance(self.t, dolfinx.Constant):
            raise RuntimeError("No stage time t as dolfinx.Constant provied.")

        if not isinstance(self.dt, dolfinx.Constant):
            raise RuntimeError("No time step dt as dolfinx.Constant provied.")

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

        # Pointers to solution states
        if "x" in kwargs:
            self.x = kwargs["x"]
        if "x0" in kwargs:
            self.x0 = kwargs["x0"]
        if "x0t" in kwargs:
            self.x0t = kwargs["x0t"]

        if "verbose" in kwargs and kwargs["verbose"] is True:
            str_alpha_f = f"alpha_f = {self.alpha_f.value:.3f}"
            str_alpha_m = f"alpha_m = {self.alpha_m.value:.3f}"
            str_gamma = f"gamma = {self.gamma.value:.3f}"
            print(f"ODEInt (generalised alpha) using:\n{str_alpha_f:s}\n{str_alpha_m:s}\n{str_gamma:s}")

    # def g_(self, g, x=None, x0=None, x0t=None):
    #     if g is None:
    #         raise RuntimeError("No function or expression given.")
    #     if x is None:
    #         x = self.x
    #     if x0 is None:
    #         x0 = self.x0
    #     if x0t is None:
    #         x0t = self.x0t

    #     # TODO: Rethink wrt non-constant expressions involving time derivative

    #     g_x = g(x, time_instant=1)
    #     g_x0 = g(x0, time_instant=0)
    #     g_x0t = g(x0t, time_instant=0)

    #     # Local function to compute the expression
    #     def _compute(g_x, g_x0, g_x0t):
    #         g_xt = 1.0 / (self.gamma * self.dt) * (g_x - g_x0) \
    #             + (self.gamma - 1.0) / self.gamma * g_x0t
    #         return self.alpha_m * g_xt + (1.0 - self.alpha_m) * g_x0t

    #     if isinstance(g_x, list) and isinstance(g_x0, list) and isinstance(g_x0t, list):
    #         # Check dimensions
    #         assert(len(g_x) == len(g_x0))
    #         assert(len(g_x) == len(g_x0t))
    #         # Return list of forms version
    #         return [_compute(_g_x, _g_x0, _g_x0t) for _g_x, _g_x0, _g_x0t in zip(g_x, g_x0, g_x0t)]
    #     else:
    #         # return form version
    #         return _compute(g_x, g_x0, g_x0t)

    # def f_(self, f, x=None, x0=None):
    #     if f is None:
    #         raise RuntimeError("No function or expression given.")
    #     if x is None:
    #         x = self.x
    #     if x0 is None:
    #         x0 = self.x0

    #     f_x = f(x, time_instant=1)
    #     f_x0 = f(x0, time_instant=0)

    #     # Local function to compute the expression
    #     def _compute(f_x, f_x0):
    #         return self.alpha_f * f_x + (1.0 - self.alpha_f) * f_x0

    #     if type(f_x) is list and type(f_x0) is list:
    #         # Check dimensions
    #         assert(len(f_x) == len(f_x0))
    #         # Return list of forms version
    #         return [_compute(_f_x, _f_x0) for _f_x, _f_x0 in zip(f_x, f_x0)]
    #     else:
    #         # Return form version
    #         return _compute(f_x, f_x0)

    def stage(self):

        self.t.value += self.alpha_f.value * self.dt.value

        return self.t, self.dt

    def eval_rate(self, x=None, x0=None, x0t=None):
        if x is None:
            x = self.x
        if x0 is None:
            x0 = self.x0
        if x0t is None:
            x0t = self.x0t

        return ((x - x0) / self.dt + (self.gamma - 1.0) * x0t) / self.gamma

    def update(self, x=None, x0=None, x0t=None):
        if x is None:
            x = self.x
        if x0 is None:
            x0 = self.x0
        if x0t is None:
            x0t = self.x0t

        self.t.value += (1.0 - self.alpha_f.value) * self.dt.value

        # update x0t
        if isinstance(x, list):
            for i, xi in enumerate(x):
                self.eval_rate(xi.vector, x0[i].vector, x0t[i].vector).copy(x0t[i].vector)
        else:
            self.eval_rate(x.vector, x0.vector, x0t.vector).copy(x0t.vector)

        # update x0
        if isinstance(x, list):
            for i, xi in enumerate(x):
                with xi.vector.localForm() as locxi, x0[i].vector.localForm() as locx0:
                    locxi.copy(locx0)
        else:
            with x.vector.localForm() as locx, x0.vector.localForm() as locx0:
                locx.copy(locx0)

        return x0, x0t

    def integrate(self, x=None, x0=None):
        if x is None:
            x = self.x
        if x0 is None:
            x0 = self.x0

        # trapezoidal rule to integrate state
        return 0.5 * self.dt * (x0 + x)
        # optional?: use integrated hermite polynomial
        # return 1./2.  *  self.dt     * ( x0  + x  ) + \
        #        1./12. * (self.dt)**2 * ( x0t - xt )

    def discretise_in_time(self, f, x, xt, x0=None, x0t=None):

        if x0 is None:
            x0 = self.x0
        if x0t is None:
            x0t = self.x0t

        xa = self.alpha_f * x + (1.0 - self.alpha_f) * x0
        xat = self.alpha_m * xt + (1.0 - self.alpha_m) * x0t

        f = dolfiny.expression.evaluate(f, x, xa)
        f = dolfiny.expression.evaluate(f, xt, xat)

        x1t = self.eval_rate(x, x0, x0t)

        f = dolfiny.expression.evaluate(f, xt, x1t)

        return f

    # @staticmethod
    # def form_hook(fn):
    #     """Decorator for hook functions to be used with ODEInt.

    #        Clears the "time_instant" function parameter in `ODEInt.g_` and `ODEInt.f_`
    #        if not needed in the implementation of the hook function.
    #     """
    #     import functools

    #     @functools.wraps(fn)
    #     def wrapper(*args, **kwargs):
    #         # Get names of parameters in decorated function fn
    #         fn_pnames = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    #         # Remove "time_instant" parameter from call kwargs if not supported by fn
    #         if "time_instant" in kwargs and "time_instant" not in fn_pnames:
    #             kwargs.pop("time_instant")
    #         # Call fn with cured kwargs
    #         return fn(*args, **kwargs)

    #     return wrapper
