from dolfinx.cpp.fem import Form_float64, Form_complex128
from petsc4py import PETSc
import numpy as np
import numba
import cffi
import itertools


ffi = cffi.FFI()
c_signature = numba.types.void(
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.uint8))
Form = Form_float64 if PETSc.ScalarType == np.float64 else Form_complex128


class LocalSolver:
    def __init__(self, function_spaces, local_spaces_id, F_integrals, J_integrals, local_integrals):
        """Local solver used to eliminate local degrees-of-freedom."""
        self.function_spaces = function_spaces
        self.local_spaces_id = local_spaces_id
        self.J_integrals = J_integrals
        self.F_integrals = F_integrals
        self.local_integrals = local_integrals

        self.global_spaces_id = [i for i in range(len(function_spaces)) if i not in self.local_spaces_id]

    def reduced_F_forms(self, F_ufc, J_ufc, F_ufl, J_ufl):
        """Return list of forms used to assemble reduced residuals"""

        # Stack all compiled and UFL forms
        forms_ufc = F_ufc
        forms_ufc += list(itertools.chain(*J_ufc))
        forms_ufc = [form.ufcx_form for form in forms_ufc if form is not None]

        forms_ufl = F_ufl
        forms_ufl += list(itertools.chain(*J_ufl))
        forms_ufl = [form for form in forms_ufl if form is not None]

        coefficients, constants = self._stack_data(forms_ufc, forms_ufl)
        print(constants)

        F_form = []

        for gi, i in enumerate(self.global_spaces_id):
            V = self.function_spaces[i]
            integrals = self.F_integrals[gi]
            for celltype, integral in integrals.items():
                itg = integral[0]
                for k in range(len(itg)):
                    itg[k] = (itg[k][0], self.wrap_kernel(itg[k][1], (i,), F_ufc, J_ufc, celltype).address)

            cppform = Form([V._cpp_object], integrals, coefficients, constants, False, None)
            F_form += [cppform]

        return F_form

    def reduced_J_forms(self, F_ufc, J_ufc, F_ufl, J_ufl):
        """Return list of list of forms used to assemble reduced tangents"""

        # Stack all compiled and UFL forms
        forms_ufc = F_ufc
        forms_ufc += list(itertools.chain(*J_ufc))
        forms_ufc = [form.ufcx_form for form in forms_ufc if form is not None]

        forms_ufl = F_ufl
        forms_ufl += list(itertools.chain(*J_ufl))
        forms_ufl = [form for form in forms_ufl if form is not None]

        coefficients, constants = self._stack_data(forms_ufc, forms_ufl)

        J_form = [[None for j in range(len(self.global_spaces_id))] for i in range(len(self.global_spaces_id))]

        for gi, i in enumerate(self.global_spaces_id):
            V0 = self.function_spaces[i]
            for gj, j in enumerate(self.global_spaces_id):
                V1 = self.function_spaces[j]
                integrals = self.J_integrals[gi][gj]
                for celltype, integral in integrals.items():
                    itg = integral[0]
                    for k in range(len(itg)):
                        itg[k] = (itg[k][0], self.wrap_kernel(itg[k][1], (i, j), F_ufc, J_ufc, celltype).address)

                J_form[gi][gj] = Form([V0._cpp_object, V1._cpp_object], integrals, coefficients, constants, False, None)

        return J_form

    def local_form(self, F_ufc, J_ufc, F_ufl, J_ufl):
        """Return list of forms used to 'assemble' into local state vectors."""

        # Stack all compiled and UFL forms
        forms_ufc = F_ufc
        forms_ufc += list(itertools.chain(*J_ufc))
        forms_ufc = [form.ufcx_form for form in forms_ufc if form is not None]

        forms_ufl = F_ufl
        forms_ufl += list(itertools.chain(*J_ufl))
        forms_ufl = [form for form in forms_ufl if form is not None]

        coefficients, constants = self._stack_data(forms_ufc, forms_ufl)

        local_form = []
        for li, i in enumerate(self.local_spaces_id):
            V = self.function_spaces[i]
            assert(self.function_spaces[li] == V)

            integrals = self.local_integrals[li]
            for celltype, integral in integrals.items():
                itg = integral[0]
                for k in range(len(itg)):
                    itg[k] = (itg[k][0], self.wrap_kernel(itg[k][1], (i,), F_ufc, J_ufc, celltype).address)

            cppform = Form([V._cpp_object], integrals, coefficients, constants, False, None)
            local_form += [cppform]

        return local_form

    def wrap_kernel(self, kernel, indices, F, J, celltype):

        @numba.cfunc(c_signature, nopython=True)
        def do_nothing(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
            pass

        sizes = []
        for i, fs in enumerate(self.function_spaces):
            sizes += [fs.element.space_dimension]

        do_nothing_cffi = ffi.cast(
            "void(*)(double *, double *, double *, double *, int *, uint8_t *)", do_nothing.address)

        sizes = np.array(sizes, dtype=int)
        # Numba does not support "list" of fn pointers, must be "tuple"
        F_fn = tuple(F[i].ufcx_form.integrals(celltype)[0].tabulate_tensor_float64 if F[i].ufcx_form.num_integrals(
            celltype) > 0 else do_nothing_cffi for i in range(len(sizes)))
        J_fn = tuple(tuple(
            J[i][j].ufcx_form.integrals(celltype)[0].tabulate_tensor_float64 if
            (J[i][j] is not None and J[i][j].ufcx_form.num_integrals(celltype) > 0) else
            do_nothing_cffi for j in range(len(sizes))) for i in range(len(sizes)))

        num_rows = sizes[indices[0]]
        num_cols = 1
        if len(indices) == 2:
            num_cols = sizes[indices[1]]

        @numba.cfunc(c_signature, nopython=True)
        def wrapped_kernel(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):

            A = numba.carray(A_, (num_rows, num_cols), dtype=PETSc.ScalarType)

            # Allocate empty data arrays for all residuals
            F_arr = [np.zeros((sizes[i], 1), dtype=PETSc.ScalarType) for i in range(len(sizes))]
            F_fn

            for i in range(len(sizes)):
                # Prepare pre-evaluated functions
                @numba.njit
                def F_fn_wrapped(w):
                    F_fn[i](ffi.from_buffer(F_arr[i]), w_, c_, coords_, entity_local_index, permutation)


            # Allocate empty data arrays for all tangents
            J_arr = [[np.zeros((sizes[i], sizes[j]), dtype=PETSc.ScalarType)
                      for j in range(len(sizes))] for i in range(len(sizes))]

            # for i in range(len(sizes)):
            #     for j in range(len(sizes)):
            #         fn = J_fn[i]
            #         fn[j](ffi.from_buffer(J_arr[i][j]), w_, c_, coords_, entity_local_index, permutation)

            # Execute user kernel
            kernel(J_fn, F_fn_wrapped, J_arr, F_arr, (w_, c_, coords_, entity_local_index, permutation), A)

        return wrapped_kernel

    def _stack_data(self, forms_ufc, forms_ufl):
        """Stack all Coefficient functions and Constants across all blocks in F and J forms"""
        coefficients = []
        constants = []
        for i, form_ufl in enumerate(forms_ufl):
            original_coefficients = form_ufl.coefficients()
            for j in range(forms_ufc[i].num_coefficients):
                coeff = original_coefficients[forms_ufc[i].original_coefficient_position[j]]._cpp_object
                if coeff not in coefficients:
                    coefficients.append(coeff)
            for const in form_ufl.constants():
                const = const._cpp_object
                if const not in constants:
                    constants.append(const)
        return coefficients, constants
