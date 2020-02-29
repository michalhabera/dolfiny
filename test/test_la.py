import numpy as np

import dolfinx
import dolfiny.la
import pytest
import scipy.sparse
from petsc4py import PETSc

skip_in_parallel = pytest.mark.skipif(
    dolfinx.MPI.size(dolfinx.MPI.comm_world) > 1,
    reason="This test should only be run in serial.")


@skip_in_parallel
def test_scipy_to_petsc():
    A = scipy.sparse.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    A_petsc = dolfiny.la.scipy_to_petsc(A)
    assert np.isclose(A_petsc.getValue(0, 1), 2.0)


@skip_in_parallel
def test_petsc_to_scipy():
    A = PETSc.Mat().createAIJ(size=(2, 2))
    A.setUp()
    A.setValuesCSR((0, 2, 4), (0, 1, 0, 1), (1.0, 2.0, 3.0, 4.0))
    A.assemble()

    A_scipy = dolfiny.la.petsc_to_scipy(A)
    assert np.isclose(A_scipy[0, 1], 2.0)
