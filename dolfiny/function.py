import ufl
import dolfinx
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
import typing
from petsc4py import PETSc


class Replacer(MultiFunction):
    def __init__(self, mapping):
        MultiFunction.__init__(self)
        self._mapping = mapping

    def expr(self, o, *args):
        if o in self._mapping:
            return self._mapping[o]
        else:
            return self.reuse_if_untouched(o, *args)


def extract_blocks(form, test_functions: typing.List[ufl.Argument],
                   trial_functions: typing.List[ufl.Argument] = None):
    """Extract blocks from a monolithic UFL form.

    Parameters
    ----------
    form
    test_functions
    trial_functions: optional

    Returns
    -------
    Splitted UFL form in the order determined by the passed test and trial functions.
    If no `trial_functions` are provided returns a list, othwerwise returns list of lists.

    """
    # Prepare empty block matrices list
    if trial_functions is not None:
        blocks = [[None for i in range(len(test_functions))] for j in range(len(trial_functions))]
    else:
        blocks = [None for i in range(len(test_functions))]

    for i, tef in enumerate(test_functions):

        if trial_functions is not None:
            for j, trf in enumerate(trial_functions):
                to_null = dict()

                # Dictionary mapping the other trial functions
                # to zero
                for item in trial_functions:
                    if item != trf:
                        to_null[item] = ufl.zero(item.ufl_shape)

                # Dictionary mapping the other test functions
                # to zero
                for item in test_functions:
                    if item != tef:
                        to_null[item] = ufl.zero(item.ufl_shape)

                replacer = Replacer(to_null)
                blocks[i][j] = map_integrand_dags(replacer, form)
        else:
            to_null = dict()

            # Dictionary mapping the other test functions
            # to zero
            for item in test_functions:
                if item != tef:
                    to_null[item] = ufl.zero(item.ufl_shape)

            replacer = Replacer(to_null)
            blocks[i] = map_integrand_dags(replacer, form)

    return blocks


def functions_to_vec(u: typing.List[dolfinx.Function], x):
    """Copies functions into block vector"""
    if x.getType() == "nest":
        for i, subvec in enumerate(x.getNestSubVecs()):
            u[i].vector.copy(subvec)
            subvec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    else:
        offset = 0
        for i in range(len(u)):
            size_local = u[i].vector.getLocalSize()
            x[offset:offset + size_local] = u[i].vector.array
            offset += size_local
            x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


def vec_to_functions(x, u: typing.List[dolfinx.Function]):
    """Copies block vector into functions"""
    if x.getType() == "nest":
        for i, subvec in enumerate(x.getNestSubVecs()):
            subvec.copy(u[i].vector)
            u[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    else:
        offset = 0
        x = x.getArray(readonly=True)
        for i in range(len(u)):
            size_local = u[i].vector.getLocalSize()
            u[i].vector.array[:] = x[offset:offset + size_local]
            offset += size_local
            u[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
