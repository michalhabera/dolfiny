
class odeint():

    def __init__(self, **kwargs):
        r"""Initialises the ODE integrator (single-step-method).
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
                alpha_f = 1/(1+rho),
                alpha_m = 1/2*(3-rho)/(1+rho),
                gamma = 1/2 + alpha_m - alpha_f
        Args:
            **kwargs:
                dt: Time step size.
                rho: Spectral radius rho_infinity for generalised alpha.
                alpha_f:
                alpha_m:
                gamma:
        """
        # Eval settings

        # Time step size
        if 'dt' in kwargs:
            self.dt = kwargs['dt']
        else:
            raise ArgumentError("No time step dt given.")

        # Default parameters (Backward-Euler)
        self.alpha_f = 1.0
        self.alpha_m = 0.5
        self.gamma   = 0.5

        # Parameters from given rho
        if 'rho' in kwargs:
            self.rho = kwargs['rho']
            self.alpha_f = 1.0 / (1.0 + self.rho)
            self.alpha_m = 0.5 * (3.0 - self.rho)/(1.0 + self.rho)
            self.gamma   = 0.5 + self.alpha_m - self.alpha_f

        # Parameters directly
        if 'alpha_f' in kwargs and 'alpha_m' in kwargs and 'gamma' in kwargs:
            self.alpha_f = kwargs['alpha_f']
            self.alpha_m = kwargs['alpha_m']
            self.gamma   = kwargs['gamma']

        # pointers to solution states (not documented)
        if 'x' in kwargs:
            self.x = kwargs['x']
        if 'x0' in kwargs:
            self.x_last_time = kwargs['x0']
        if 'x0t' in kwargs:
            self.dxdt_last_time = kwargs['x0t']

        if 'verbose' in kwargs and kwargs['verbose'] is True:
            out = "rho = %.3f, alpha_f = %.3f, alpha_m = %.3f, gamma = %.3f" % \
                  (self.rhoinf, self.alpha_f, self.alpha_m, self.gamma)
            print("odeint (generalised alpha) using: %s" % out )

    def g_(self, g, x=None, x0=None, x0t=None):
        if g is None: raise ArgumentError("No function or expression given.")
        if x is None: x=self.x
        if x0 is None: x0=self.x_last_time
        if x0t is None: x0t=self.dxdt_last_time

        # TODO: Rethink wrt non-constant expressions involving time derivative

        g_x = g(x, time_instant=1)
        g_x0 = g(x0, time_instant=0)
        g_x0t = g(x0t, time_instant=0)

        # local function to compute the expression
        def _compute(g_x, g_x0, g_x0t):
            g_xt = 1.0/(self.gamma*self.dt) * ( g_x - g_x0 ) \
                   + (self.gamma-1.0)/self.gamma * g_x0t
            return self.alpha_m * g_xt + (1.0-self.alpha_m) * g_x0t

        if type(g_x) is list and type(g_x0) is list and type(g_x0t) is list:
            # check dimensions
            assert(len(g_x) == len(g_x0))
            assert(len(g_x) == len(g_x0t))
            # return list of forms version
            return [ _compute(_g_x, _g_x0, _g_x0t) for _g_x, _g_x0, _g_x0t in zip(g_x, g_x0, g_x0t) ]
        else:
            # return form version
            return _compute(g_x, g_x0, g_x0t)

    def f_(self, f, x=None, x0=None):
        if f is None: raise ArgumentError("No function or expression given.")
        if x is None: x=self.x
        if x0 is None: x0=self.x_last_time

        f_x = f(x, time_instant=1)
        f_x0 = f(x0, time_instant=0)

        # local function to compute the expression
        def _compute(f_x, f_x0):
            return self.alpha_f * f_x + (1.0-self.alpha_f) * f_x0

        if type(f_x) is list and type(f_x0) is list:
            # check dimensions
            assert(len(f_x) == len(f_x0))
            # return list of forms version
            return [ _compute(_f_x, _f_x0) for _f_x, _f_x0 in zip(f_x, f_x0) ]
        else:
            # return form version
            return _compute(f_x, f_x0)

    def eval_rate(self, x=None, x0=None, x0t=None):
        if x is None: x=self.x
        if x0 is None: x0=self.x_last_time
        if x0t is None: x0t=self.dxdt_last_time

        return 1.0/(self.gamma*self.dt) * (x - x0) \
               + (self.gamma-1.0)/self.gamma * x0t

    def update(self, x=None, x0=None, x0t=None):
        if x is None: x=self.x
        if x0 is None: x0=self.x_last_time
        if x0t is None: x0t=self.dxdt_last_time

        if isinstance(x, list):
            for i, xi in enumerate(x):
                xi.vector.copy(x0[i].vector)
        else:
            x.vector.copy(x0.vector)

        if isinstance(x, list):
            for i, xi in enumerate(x):
                self.eval_rate(xi.vector, x0[i].vector, x0t[i].vector).copy(x0t[i].vector)
        else:
            self.eval_rate(x.vector, x0.vector, x0t.vector).copy(x0t.vector)

        return x0, x0t

    def integrate(self, x, x0):
        # trapezoidal rule to integrate state
        return 0.5 * self.dt * ( x0 + x )
        # optional?: use integrated hermite polynomial
        # return 1./2.  *  self.dt     * ( x0  + x  ) + \
        #        1./12. * (self.dt)**2 * ( x0t - xt )
