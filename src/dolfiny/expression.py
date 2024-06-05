import dolfinx
import ufl
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.corealg.multifunction import MultiFunction


def evaluate(e, u, u0):
    """Evaluate a UFL expression (or list of expressions).
       Basically replaces function(s) u by function(s) u0.

    Parameters
    ----------
    e: UFL Expr or list of UFL expressions/forms
    u: UFL Function or list of UFL functions
    u0: UFL Function or list of UFL functions

    Returns
    -------
    Expression (or list of expressions) with function(s) u replaced by function(s) u0.

    """

    from ufl.algorithms.map_integrands import map_integrand_dags
    from ufl.algorithms.replace import Replacer

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
    e: UFL Expr or list of UFL expressions/forms
    u: UFL Function or list of UFL functions
    du: UFL Function or list of UFL functions
    u0: UFL Function or list of UFL functions, defaults to u

    Returns
    -------
    Expression (or list of expressions) with functional derivative.

    """

    if u0 is None:
        u0 = u

    e0 = evaluate(e, u, u0)

    if isinstance(e0, list):
        de0 = []
        for e0_ in e0:
            if isinstance(u0, list) and isinstance(du, list):
                de0_ = sum(ufl.derivative(e0_, v0, dv) for v0, dv in zip(u0, du))
            else:
                de0_ = ufl.derivative(e0_, u0, du)
            de0_ = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(de0_)
            de0_ = ufl.algorithms.apply_derivatives.apply_derivatives(de0_)
            de0.append(de0_)
    else:
        if isinstance(u0, list) and isinstance(du, list):
            de0 = sum(ufl.derivative(e0, v0, dv) for v0, dv in zip(u0, du))
        else:
            de0 = ufl.derivative(e0, u0, du)
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
    u0: UFL Function or list of UFL functions, defaults to zero

    Returns
    -------
    1st order Taylor series expansion of expression/form (or list of expressions/forms).

    """

    if u0 is None:
        if isinstance(u, list):
            u0 = []
            for v in u:
                u0.append(dolfinx.fem.Function(v.function_space, name=v.name + "_0"))
        else:
            u0 = dolfinx.fem.Function(u.function_space, name=u.name + "_0")

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
    """Assemble UFL form given by UFL expression e and UFL integration measure dx.
       The expression can be a tensor quantity of rank 0, 1 or 2.

    Parameters
    ----------
    e: UFL Expr
    dx: UFL Measure

    Returns
    -------
    Assembled form f = e * dx as scalar or numpy array (depends on rank of e).

    """

    from mpi4py import MPI

    import numpy as np

    rank = ufl.rank(e)
    shape = ufl.shape(e)

    if rank == 0:
        f = dolfinx.fem.form(e * dx)
        f_ = MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(f), op=MPI.SUM)
    elif rank == 1:
        f_ = np.zeros(shape)
        for row in range(shape[0]):
            f = dolfinx.fem.form(e[row] * dx)
            f_[row] = MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(f), op=MPI.SUM)
    elif rank == 2:
        f_ = np.zeros(shape)
        for row in range(shape[0]):
            for col in range(shape[1]):
                f = dolfinx.fem.form(e[row, col] * dx)
                f_[row, col] = MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(f), op=MPI.SUM)
    return f_


def extract_linear_combination(e, linear_comb=[], scalar_weight=1.0):
    """Extract linear combination from UFL expression.

    Assumes the expression could be equivalently written as ``\\sum_i c_i u_i``
    where ``c_i`` are known scalar coefficients and ``u_i`` are dolfinx Functions.
    If this assumption fails, raises a RuntimeError.

    Returns
    -------
    Tuples (u_i, c_i) which represent summands in the above sum.
    Returned summands are not uniquely accumulated, i.e. could return (u, 1.0) and (u, 2.0).

    Note
    ----
    Constant nodes (dolfinx.Constant) are not handled. So the expression which has the above
    form where ``c_i`` could contain Constant must have first these nodes numerically evaluated.

    """

    from ufl.classes import ComponentTensor, Division, Index, Indexed, Product, ScalarValue, Sum

    if isinstance(e, dolfinx.fem.Function):
        linear_comb.append((e, scalar_weight))
    elif isinstance(e, Product | Division):
        e0, e1 = e.ufl_operands

        if isinstance(e0, ScalarValue):
            scalar, expr = e0, e1
        elif isinstance(e1, ScalarValue):
            scalar, expr = e1, e0
        else:
            raise RuntimeError(f"One operand of {type(e)} must be ScalarValue")

        if isinstance(e, Product):
            scalar_weight *= float(scalar)
        else:
            scalar_weight /= float(scalar)

        extract_linear_combination(expr, linear_comb, scalar_weight)
    elif isinstance(e, Sum):
        e0, e1 = e.ufl_operands
        extract_linear_combination(e0, linear_comb, scalar_weight)
        extract_linear_combination(e1, linear_comb, scalar_weight)
    elif isinstance(e, ComponentTensor | Indexed):
        expr, indices = e.ufl_operands
        if not all(isinstance(i, Index) for i in indices):
            raise RuntimeError("Expecting free index, not fixed index")
        extract_linear_combination(expr, linear_comb, scalar_weight)
    else:
        raise RuntimeError(f"Expression type {type(e)} not handled")

    return linear_comb


class ConstantEvaluator(MultiFunction):
    expr = MultiFunction.reuse_if_untouched

    def constant(self, a):
        if a.value.shape == ():
            return float(a.value)
        else:
            return ufl.as_tensor(a.value)


def evaluate_constants(expr):
    """Transform Constant nodes into numeric UFL nodes"""
    return map_integrand_dags(ConstantEvaluator(), expr)
