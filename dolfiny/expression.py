import ufl
import dolfinx


def evaluate(e, u, u0):
    """Evaluate a UFL expression (or list of expressions).
    Basically replaces function(s) u by function(s) u0.

    Parameters
    ----------
    e: UFL Expr or list of UFL expressions
    u: UFL Function or list of UFL functions
    u0: UFL Function or list of UFL functions

    Returns
    -------
    Expr (or list of expressions) with function u replaced by function u0.

    """

    from ufl.algorithms.replace import Replacer
    from ufl.algorithms.map_integrands import map_integrand_dags

    if isinstance(u, list) and isinstance(u0, list):
        repmap = {v: v0 for v, v0 in zip(u, u0)}
    elif not isinstance(u, list) and not isinstance(u0, list):
        repmap = {u: u0}
    else:
        raise RuntimeError("Incompatible functions (list-of-functions and function) provided.")

    replacer = Replacer(repmap)

    if isinstance(e, list):
        e0 = []
        for e_ in e:
            e0.append(map_integrand_dags(replacer, e_))
    else:
        e0 = map_integrand_dags(replacer, e)

    return e0


def derivative(e, u, du, u0=None):
    """Generate the functional derivative of UFL expression (or list of expressions) for the
       given function(s) u in the direction of function(s) du and at u0.

       Example (first variation): δe = dolfiny.expression.derivative(e, m, δm)

    Parameters
    ----------
    e: UFL Expr or list of UFL expressions
    u: UFL Function or list of UFL functions
    du: UFL Function or list of UFL functions
    u0: UFL Function or list of UFL functions, defaults to u

    Returns
    -------
    Functional derivative.

    """

    if u0 is None:
        u0 = u

    e0 = evaluate(e, u, u0)

    if isinstance(e0, list):
        de0 = []
        for e0_ in e0:
            de0_ = sum(ufl.derivative(e0_, v0, du) for v0, du in zip(u0, du))
            de0_ = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(de0_)
            de0_ = ufl.algorithms.apply_derivatives.apply_derivatives(de0_)
        de0.append(de0_)
    else:
        de0 = sum(ufl.derivative(e0, v0, du) for v0, du in zip(u0, du))
        de0 = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(de0)
        de0 = ufl.algorithms.apply_derivatives.apply_derivatives(de0)

    return de0


def linearise(e, u, u0=None):
    """Generate the first order Taylor series expansion of UFL expressions (or list of expressions)
       for the given function(s) u at u0.

       Example (linearise around zero): linF = dolfiny.expression.linearise(F, u)
       Example (linearise around given state): linF = dolfiny.expression.linearise(F, u, u0)

    Parameters
    ----------
    e: UFL Expr/Form or list of UFL expressions/forms
    u: UFL Function or list of UFL functions
    du: UFL Function or list of UFL functions
    u0: UFL Function or list of UFL functions, defaults to zero

    Returns
    -------
    1st order Taylor series expansion of expression/form (or list of expressions/forms).

    """

    if u0 is None:
        # TODO: do not assume u as Function
        if isinstance(u, list):
            u0 = []
            for v in u:
                u0.append(dolfinx.Function(v.function_space, name=v.name + '_0'))
        else:
            u0 = dolfinx.Function(u.function_space, name=u.name + '_0')

    e0 = evaluate(e, u, u0)
    deu = derivative(e, u, u, u0)
    deu0 = derivative(e, u, u0, u0)

    if isinstance(e, list):
        de = []
        for e0_, deu_, deu0_ in zip(e0, deu, deu0):
            de.append(e0_ + (deu_ - deu0_))
    else:
        de = e0 + (deu - deu0)

    return de


def assemble(e, dx):
    """Assemble form given by expression e and integration measure dx.
       The expression can be a tensor quantity of rank 0, 1 or 2.

    Parameters
    ----------
    e: UFL Expr
    dx: UFL Measure

    Returns
    -------
    Assembled form f = e * dx

    """

    import numpy as np

    rank = ufl.rank(e)
    shape = ufl.shape(e)

    if rank == 0:
        f_ = dolfinx.fem.assemble_scalar(e * dx)
    elif rank == 1:
        f_ = np.zeros(shape)
        for row in range(shape[0]):
            f_[row] = dolfinx.fem.assemble_scalar(e[row] * dx)
    elif rank == 2:
        f_ = np.zeros(shape)
        for row in range(shape[0]):
            for col in range(shape[1]):
                f_[row, col] = dolfinx.fem.assemble_scalar(e[row, col] * dx)
    return f_
