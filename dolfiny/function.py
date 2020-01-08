import ufl
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
import typing


class Replacer(MultiFunction):
    def __init__(self, mapping):
        MultiFunction.__init__(self)
        self._mapping = mapping

    def expr(self, o, *args):
        if o in self._mapping:
            return self._mapping[o]
        else:
            return self.reuse_if_untouched(o, *args)


def extract_blocks(form, test_functions: typing.List, trial_functions: typing.List):
    """Extract blocks from a monolithic UFL form.

    Returns
    -------
    Splitted UFL form in the order determined by the passed test and trial functions.

    """
    # Prepare empty block matrices list
    blocks = [[None for i in range(len(test_functions))] for j in range(len(trial_functions))]

    for i, tef in enumerate(test_functions):
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

    return blocks
