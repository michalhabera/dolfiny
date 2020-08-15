import numpy as np

import cffi
import dolfinx
import ffcx.fiatinterface
import numba
import numba.core.typing.cffi_utils as cffi_support
import ufl
from numba.typed import List
from petsc4py import PETSc

# Load ffi import-time, needed to be binded
# into numba compiled code for interpolation
ffi = cffi.FFI()


class CompiledExpression:
    def __init__(self, expr, target_el):
        self.expr = expr
        self.target_el = target_el

        if target_el.value_size() > 1:
            # For mixed elements fetch only one element
            # Saves computation for vector/tensor elements,
            # no need to evaluate at same points for each vector
            # component
            #
            # TODO: We can get unique subelements or unique
            #       points for evaluation
            sub_elements = target_el.sub_elements()

            # We can handle only all sub elements equal case
            assert all([sub_elements[0] == x for x in sub_elements])
            target_el = sub_elements[0]

        # Identify points at which to evaluate the expression
        self.fiat_element = ffcx.fiatinterface.create_element(target_el)

        if not all(x == "affine" for x in self.fiat_element.mapping()):
            raise NotImplementedError("Only affine mapped function spaces supported")

        nodes = []
        for dual in self.fiat_element.dual_basis():
            pts, = dual.pt_dict.keys()
            nodes.append(pts)

        nodes = np.asarray(nodes)

        module = dolfinx.jit.ffcx_jit((expr, nodes))
        self.module = module


def interpolate(expr, target_func):
    """Compile-interpolate UFL expression.

    Note
    ----
    Works only for affine-mapped point-evaluation finite elements, e.g.
    lagrange/discontinuous lagrange of arbitrary order.

    """
    compiled_expression = CompiledExpression(expr, target_func.function_space.ufl_element())
    interpolate_cached(compiled_expression, target_func)


def interpolate_cached(compiled_expression, target_func):
    kernel = compiled_expression.module.tabulate_expression

    # Register complex types
    cffi_support.register_type(ffi.typeof('double _Complex'),
                               numba.types.complex128)
    cffi_support.register_type(ffi.typeof('float _Complex'),
                               numba.types.complex64)

    reference_geometry = np.asarray(compiled_expression.fiat_element.ref_el.get_vertices())

    # Unpack mesh and dofmap data
    mesh = target_func.function_space.mesh
    geom_dofmap = mesh.geometry.dofmap.array
    geom_pos = mesh.geometry.dofmap.offsets
    geom = mesh.geometry.x
    gdim = mesh.geometry.dim

    dofmap = target_func.function_space.dofmap.list.array

    # Prepare coefficients and their dofmaps
    # Global vectors and dofmaps are prepared here, local are
    # fetched inside hot cell-loop

    # Number of coefficients in ffcx-processed ufl form
    num_coeffs = compiled_expression.module.num_coefficients
    # Positions of ffcx-preprocessed coefficients in original form
    cpos = compiled_expression.module.original_coefficient_positions

    coeffs = ufl.algorithms.analysis.extract_coefficients(compiled_expression.expr)
    coeffs_dofmaps = List.empty_list(numba.types.Array(numba.typeof(dofmap[0]), 1, "C", readonly=True))
    coeffs_vectors = List.empty_list(numba.types.Array(numba.typeof(PETSc.ScalarType()), 1, "C", readonly=True))

    for i in range(num_coeffs):
        coeffs_dofmaps.append(coeffs[cpos[i]].function_space.dofmap.list.array)
        coeffs_vectors.append(np.asarray(coeffs[cpos[i]].vector))

    local_coeffs_sizes = np.asarray([coeff.function_space.element.space_dimension()
                                     for coeff in coeffs], dtype=np.int)
    local_coeffs_size = np.sum(local_coeffs_sizes, dtype=np.int)

    # Prepare and pack constants
    constants = ufl.algorithms.analysis.extract_constants(compiled_expression.expr)
    constants_vector = np.array([], dtype=PETSc.ScalarType())
    if len(constants) > 0:
        constants_vector = np.hstack([c.value.flatten() for c in constants])

    # Num DOFs of the target element
    local_b_size = target_func.function_space.element.space_dimension()
    value_size = int(np.product(compiled_expression.expr.ufl_shape))
    num_coeffs = len(coeffs_vectors)

    with target_func.vector.localForm() as b:
        b.set(0.0)
        assemble_vector_ufc(np.asarray(b), kernel, (geom_dofmap, geom_pos, geom), dofmap,
                            coeffs_vectors, coeffs_dofmaps, constants_vector, reference_geometry,
                            local_coeffs_sizes, local_coeffs_size, local_b_size, gdim, value_size)


@numba.njit
def assemble_vector_ufc(b, kernel, mesh, dofmap, coeffs_vectors, coeffs_dofmaps,
                        const_vector, reference_geometry, local_coeffs_sizes, local_coeffs_size,
                        local_b_size, gdim, value_size):
    geom_dofmap, geom_pos, geom = mesh

    # Coord dofs have shape (num_geometry_dofs, gdim)
    coordinate_dofs = np.zeros((geom_pos[1], gdim))
    coeffs = np.zeros(local_coeffs_size, dtype=PETSc.ScalarType)
    b_local = np.zeros(local_b_size, dtype=PETSc.ScalarType)
    dofs_per_block = int(local_b_size / value_size)

    for i, cell in enumerate(geom_pos[:-1]):
        num_vertices = geom_pos[i + 1] - geom_pos[i]
        c = geom_dofmap[cell:cell + num_vertices]
        for j in range(geom_pos[1]):
            for k in range(gdim):
                coordinate_dofs[j, k] = geom[c[j], k]
        b_local.fill(0.0)

        offset = 0
        for j in range(len(coeffs_vectors)):
            local_dofsize = local_coeffs_sizes[j]
            coeffs[offset:offset + local_dofsize] = coeffs_vectors[j][coeffs_dofmaps[j]
                                                                      [i * local_dofsize:local_dofsize * (i + 1)]]
            offset += local_dofsize

        kernel(ffi.from_buffer(b_local), ffi.from_buffer(coeffs),
               ffi.from_buffer(const_vector), ffi.from_buffer(coordinate_dofs))

        for j in range(dofs_per_block):
            for k in range(value_size):
                b[dofmap[i * local_b_size + j * value_size + k]] = b_local[dofs_per_block * k + j]
