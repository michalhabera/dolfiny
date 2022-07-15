from dolfinx.cpp.fem import Form_float64, Form_complex128
from petsc4py import PETSc
import numpy as np
import numba
import cffi
import itertools
import collections


ffi = cffi.FFI()
c_signature = numba.types.void(
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.uint8))
Form = Form_float64 if PETSc.ScalarType == np.float64 else Form_complex128


KernelData = collections.namedtuple("KernelData", ("kernel", "array", "w", "c", "coords"))
CoefficientData = collections.namedtuple("CoefficientData", ("indices", "name", "begin", "end"))
ConstantData = collections.namedtuple("ConstantData", ("indices", "name", "begin", "end"))


from numba.core import extending, cgutils, types


@extending.intrinsic
def printf(typingctx, format_type, *args):
    """printf that can be called from Numba jit-decorated functions.
    """
    if isinstance(format_type, types.StringLiteral):
        sig = types.void(format_type, types.BaseTuple.from_types(args))

        def codegen(context, builder, signature, args):
            cgutils.printf(builder, format_type.literal_value, *args[1:])
        return sig, codegen


class LocalSolver:
    def __init__(self, function_spaces, local_spaces_id, F_integrals, J_integrals, local_integrals):
        """Local solver used to eliminate local degrees-of-freedom."""
        self.function_spaces = function_spaces
        self.local_spaces_id = local_spaces_id
        self.J_integrals = J_integrals
        self.F_integrals = F_integrals
        self.local_integrals = local_integrals

        self.global_spaces_id = [i for i in range(len(function_spaces)) if i not in self.local_spaces_id]

        self.F_ufc = None
        self.J_ufc = None
        self.F_ufl = None
        self.J_ufl = None

    def reduced_F_forms(self):
        """Return list of forms used to assemble reduced residuals"""

        coefficients, constants, _, _ = self._stack_data()

        F_form = []

        for gi, i in enumerate(self.global_spaces_id):
            V = self.function_spaces[i]
            integrals = self.F_integrals[gi]
            for celltype, integral in integrals.items():
                itg = integral[0]
                for k in range(len(itg)):
                    itg[k] = (itg[k][0], self.wrap_kernel(itg[k][1], (i,), celltype).address)

            cppform = Form([V._cpp_object], integrals, [c[0]
                           for c in coefficients], [c[0] for c in constants], False, None)
            F_form += [cppform]

        return F_form

    def reduced_J_forms(self):
        """Return list of list of forms used to assemble reduced tangents"""

        coefficients, constants, _, _ = self._stack_data()

        J_form = [[None for j in range(len(self.global_spaces_id))] for i in range(len(self.global_spaces_id))]

        for gi, i in enumerate(self.global_spaces_id):
            V0 = self.function_spaces[i]
            for gj, j in enumerate(self.global_spaces_id):
                V1 = self.function_spaces[j]
                integrals = self.J_integrals[gi][gj]
                for celltype, integral in integrals.items():
                    itg = integral[0]
                    for k in range(len(itg)):
                        itg[k] = (itg[k][0], self.wrap_kernel(itg[k][1], (i, j), celltype).address)

                J_form[gi][gj] = Form([V0._cpp_object, V1._cpp_object], integrals, [c[0]
                                      for c in coefficients], [c[0] for c in constants], False, None)

        return J_form

    def local_form(self):
        """Return list of forms used to 'assemble' into local state vectors."""

        coefficients, constants, _, _ = self._stack_data()

        local_form = []
        for li, i in enumerate(self.local_spaces_id):
            V = self.function_spaces[i]
            assert(self.function_spaces[li] == V)

            integrals = self.local_integrals[li]
            for celltype, integral in integrals.items():
                itg = integral[0]
                for k in range(len(itg)):
                    itg[k] = (itg[k][0], self.wrap_kernel(itg[k][1], (i,), celltype).address)

            cppform = Form([V._cpp_object], integrals, [c[0]
                           for c in coefficients], [c[0] for c in constants], False, None)
            local_form += [cppform]

        return local_form

    def wrap_kernel(self, kernel, indices, celltype):

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
        F_fn = tuple(self.F_ufc[i].ufcx_form.integrals(celltype)[0].tabulate_tensor_float64
                     if self.F_ufc[i].ufcx_form.num_integrals(
            celltype) > 0 else do_nothing_cffi for i in range(len(sizes)))
        J_fn = tuple(tuple(
            self.J_ufc[i][j].ufcx_form.integrals(celltype)[0].tabulate_tensor_float64 if
            (self.J_ufc[i][j] is not None and self.J_ufc[i][j].ufcx_form.num_integrals(celltype) > 0) else
            do_nothing_cffi for j in range(len(sizes))) for i in range(len(sizes)))

        shape = (sizes[indices[0]], 1)
        if len(indices) == 2:
            shape = (sizes[indices[0]], sizes[indices[1]])

        stacked_coefficients, stacked_constants, coefficients, constants = self._stack_data()

        stacked_coefficients_size = stacked_coefficients[-1][1][1]
        stacked_constants_size = stacked_constants[-1][1][1]

        num_coordinate_dofs = self.function_spaces[0].mesh.geometry.dofmap.offsets[1]

        @numba.cfunc(c_signature, nopython=False)
        def wrapped_kernel(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):

            A = numba.carray(A_, shape, dtype=PETSc.ScalarType)
            w = numba.carray(w_, (stacked_coefficients_size, ), dtype=PETSc.ScalarType)
            c = numba.carray(c_, (stacked_constants_size, ), dtype=PETSc.ScalarType)
            coords = numba.carray(coords_, (num_coordinate_dofs * 3, ), dtype=np.double)

            J = numba.typed.List()
            F = numba.typed.List()

            for i in range(len(sizes)):
                F_array = np.zeros((sizes[i], 1), dtype=PETSc.ScalarType)

                coeffs = [coeff for coeff in coefficients if coeff.indices == (i, -1)]
                consts = [const for const in constants if const.indices == (i, -1)]
                w_array = np.zeros((coeffs[-1].end if len(coeffs) > 0 else 1, ), dtype=PETSc.ScalarType)
                c_array = np.zeros((consts[-1].end if len(consts) > 0 else 1, ), dtype=PETSc.ScalarType)

                # Copy coefficient data to local vector for this kernel
                pos = 0
                for coeff in coeffs:
                    size = coeff.end - coeff.begin
                    w_array[pos:pos + size] = w[coeff.begin:coeff.end]
                    pos += size

                pos = 0
                for const in consts:
                    size = const.end - const.begin
                    c_array[pos:pos + size] = c[const.begin:const.end]
                    pos += size

                F_fn[i](ffi.from_buffer(F_array), ffi.from_buffer(w_array), ffi.from_buffer(
                    c_array), ffi.from_buffer(coords), entity_local_index, permutation)

                F.append(KernelData(F_fn[i], F_array, w, c, coords))

                J_row = numba.typed.List()
                for j in range(len(sizes)):

                    coeffs = [coeff for coeff in coefficients if coeff.indices == (i, j)]
                    consts = [const for const in constants if const.indices == (i, j)]
                    w_array = np.zeros((coeffs[-1].end if len(coeffs) > 0 else 1, ), dtype=PETSc.ScalarType)
                    c_array = np.zeros((consts[-1].end if len(consts) > 0 else 1, ), dtype=PETSc.ScalarType)

                    # Copy coefficient data to local vector for this kernel
                    pos = 0
                    for coeff in coeffs:
                        size = coeff.end - coeff.begin
                        w_array[pos:pos + size] = w[coeff.begin:coeff.end]
                        pos += size

                    pos = 0
                    for const in consts:
                        size = const.end - const.begin
                        c_array[pos:pos + size] = c[const.begin:const.end]
                        pos += size

                    J_array = np.zeros((sizes[i], sizes[j]), dtype=PETSc.ScalarType)
                    J_fn[i][j](ffi.from_buffer(J_array), ffi.from_buffer(w_array), ffi.from_buffer(
                        c_array), ffi.from_buffer(coords), entity_local_index, permutation)
                    J_row.append(KernelData(J_fn[i][j], J_array, w_array, c_array, coords))
                J.append(J_row)

            # Execute user kernel
            kernel(A, J, F, entity_local_index, permutation)

        return wrapped_kernel

    def _stack_data(self):
        """Stack all Coefficients and Constants across all blocks in F and J forms"""
        stacked_coefficients = []
        stacked_constants = []
        coefficients = []
        constants = []

        # Stack all compiled and UFL forms
        stacked_forms_ufc = self.F_ufc.copy()
        stacked_forms_ufc += list(itertools.chain(*self.J_ufc))
        stacked_forms_ufc = [form.ufcx_form for form in stacked_forms_ufc if form is not None]

        stacked_forms_ufl = self.F_ufl.copy()
        stacked_forms_ufl += list(itertools.chain(*self.J_ufl))
        stacked_forms_ufl = [form for form in stacked_forms_ufl if form is not None]

        size = 0
        size_const = 0

        for i, form_ufl in enumerate(stacked_forms_ufl):
            original_coefficients = form_ufl.coefficients()
            for j in range(stacked_forms_ufc[i].num_coefficients):
                coeff = original_coefficients[stacked_forms_ufc[i].original_coefficient_position[j]]._cpp_object
                if coeff not in [c[0] for c in stacked_coefficients]:
                    stacked_coefficients.append((coeff, (size, size + coeff.function_space.element.space_dimension)))
                    size += coeff.function_space.element.space_dimension
            for const in form_ufl.constants():
                const = const._cpp_object
                if const not in [c[0] for c in stacked_constants]:
                    stacked_constants.append((const, (size_const, size_const + const.value.size)))
                    size_const += const.value.size

        for i in range(len(self.F_ufl)):

            original_coefficients = self.F_ufl[i].coefficients()
            for c in range(self.F_ufc[i].ufcx_form.num_coefficients):
                coeff = original_coefficients[self.F_ufc[i].ufcx_form.original_coefficient_position[c]]._cpp_object
                coeff_pos = [(c[1][0], c[1][1]) for c in stacked_coefficients if c[0] == coeff]
                dat = CoefficientData((i, -1), coeff.name, coeff_pos[0][0], coeff_pos[0][1])
                coefficients.append(dat)
            for const in self.F_ufl[i].constants():
                const_pos = [(c[1][0], c[1][1]) for c in stacked_constants if c[0] == const._cpp_object]
                dat = ConstantData((i, -1), str(const), const_pos[0][0], const_pos[0][1])
                constants.append(dat)

            for j in range(len(self.F_ufl)):
                original_coefficients = self.J_ufl[i][j].coefficients()
                for c in range(self.J_ufc[i][j].ufcx_form.num_coefficients):
                    coeff = original_coefficients[self.J_ufc[i]
                                                  [j].ufcx_form.original_coefficient_position[c]]._cpp_object
                    coeff_pos = [(c[1][0], c[1][1]) for c in stacked_coefficients if c[0] == coeff]
                    dat = CoefficientData((i, j), coeff.name, coeff_pos[0][0], coeff_pos[0][1])
                    coefficients.append(dat)
                for const in self.J_ufl[i][j].constants():
                    const_pos = [(c[1][0], c[1][1]) for c in stacked_constants if c[0] == const._cpp_object]
                    dat = ConstantData((i, j), str(const), const_pos[0][0], const_pos[0][1])
                    constants.append(dat)

        return stacked_coefficients, stacked_constants, tuple(coefficients), tuple(constants)
