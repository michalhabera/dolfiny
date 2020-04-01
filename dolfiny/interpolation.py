import numpy as np

import cffi
import dolfinx
import ffcx
import numba
import numba.cffi_support
import ufl
from numba.typed import List
from petsc4py import PETSc

# Load ffi import-time, needed to be binded
# into numba compiled code for interpolation
ffi = cffi.FFI()


def interpolate(expr, target_func):
    """Compile-interpolate UFL expression.

    Note
    ----
    Works only for affine-mapped point-evaluation finite elements, e.g.
    lagrange/discontinuous lagrange of arbitrary order.

    """

    target_el = target_func.function_space.ufl_element()

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
    fiat_element = ffcx.fiatinterface.create_element(target_el)

    if not all(x == "affine" for x in fiat_element.mapping()):
        raise NotImplementedError("Only affine mapped function spaces supported")

    nodes = []
    for dual in fiat_element.dual_basis():
        pts, = dual.pt_dict.keys()
        nodes.append(pts)

    nodes = np.asarray(nodes)

    module = dolfinx.jit.ffcx_jit((expr, nodes))
    kernel = module.tabulate_expression

    # Register complex types
    numba.cffi_support.register_type(ffi.typeof('double _Complex'),
                                     numba.types.complex128)
    numba.cffi_support.register_type(ffi.typeof('float _Complex'),
                                     numba.types.complex64)

    reference_geometry = np.asarray(fiat_element.ref_el.get_vertices())

    # Unpack mesh and dofmap data
    mesh = target_func.function_space.mesh
    geom_dofmap = mesh.geometry.dofmap().array()
    geom_pos = mesh.geometry.dofmap().offsets()
    geom = mesh.geometry.x

    dofmap = target_func.function_space.dofmap.list.array()

    # Prepare coefficients and their dofmaps
    # Global vectors and dofmaps are prepared here, local are
    # fetched inside hot cell-loop

    # Number of coefficients in ffcx-processed ufl form
    num_coeffs = module.num_coefficients
    # Positions of ffcx-preprocessed coefficients in original form
    cpos = module.original_coefficient_positions

    coeffs = ufl.algorithms.analysis.extract_coefficients(expr)
    coeffs_dofmaps = List.empty_list(numba.types.Array(numba.typeof(dofmap[0]), 1, "C", readonly=True))
    coeffs_vectors = List.empty_list(numba.types.Array(numba.typeof(PETSc.ScalarType()), 1, "C", readonly=True))

    for i in range(num_coeffs):
        coeffs_dofmaps.append(coeffs[cpos[i]].function_space.dofmap.list.array())
        coeffs_vectors.append(np.asarray(coeffs[cpos[i]].vector))

    local_coeffs_sizes = np.asarray([coeff.function_space.element.space_dimension()
                                     for coeff in coeffs], dtype=np.int)
    local_coeffs_size = np.sum(local_coeffs_sizes, dtype=np.int)

    # Prepare and pack constants
    constants = ufl.algorithms.analysis.extract_constants(expr)
    constants_vector = np.array([], dtype=PETSc.ScalarType())
    if len(constants) > 0:
        constants_vector = np.hstack([c.value.flatten() for c in constants])

    # Num DOFs of the target element
    local_b_size = target_func.function_space.element.space_dimension()
    num_coeffs = len(coeffs_vectors)

    with target_func.vector.localForm() as b:
        b.set(0.0)
        assemble_vector_ufc(np.asarray(b), kernel, (geom_dofmap, geom_pos, geom), dofmap,
                            coeffs_vectors, coeffs_dofmaps, constants_vector, reference_geometry,
                            local_coeffs_sizes, local_coeffs_size, local_b_size)


@numba.njit
def assemble_vector_ufc(b, kernel, mesh, dofmap, coeffs_vectors, coeffs_dofmaps,
                        const_vector, reference_geometry, local_coeffs_sizes, local_coeffs_size, local_b_size):
    geom_dofmap, geom_pos, geom = mesh
    coordinate_dofs = np.zeros(reference_geometry.shape)
    coeffs = np.zeros(local_coeffs_size, dtype=PETSc.ScalarType)
    b_local = np.zeros(local_b_size, dtype=PETSc.ScalarType)

    for i, cell in enumerate(geom_pos[:-1]):
        num_vertices = geom_pos[i + 1] - geom_pos[i]
        c = geom_dofmap[cell:cell + num_vertices]
        for j in range(reference_geometry.shape[0]):
            for k in range(reference_geometry.shape[1]):
                coordinate_dofs[j, k] = geom[c[j], k]
        b_local.fill(0.0)

        offset = 0
        for j in range(len(coeffs_vectors)):
            local_dofsize = local_coeffs_sizes[j]
            for k in range(local_dofsize):
                coeffs[offset + k] = coeffs_vectors[j][coeffs_dofmaps[j][i * local_dofsize + k]]
            offset += local_dofsize

        kernel(ffi.from_buffer(b_local), ffi.from_buffer(coeffs),
               ffi.from_buffer(const_vector), ffi.from_buffer(coordinate_dofs))

        for j in range(local_b_size):
            b[dofmap[i * local_b_size + j]] = b_local[j]
