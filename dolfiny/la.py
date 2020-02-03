from scipy.sparse import csr_matrix
from petsc4py import PETSc


def petsc_to_scipy(A):
    """Converts PETSc serial matrix to SciPy CSR matrix"""
    ai, aj, av = A.getValuesCSR()
    mat = csr_matrix((av, aj, ai))

    return mat


def scipy_to_petsc(A):
    """Converts SciPy CSR matrix to PETSc serial matrix."""
    nrows = A.shape[0]
    ncols = A.shape[1]

    ai, aj, av = A.indptr, A.indices, A.data
    mat = PETSc.Mat()
    mat.createAIJ(size=(nrows, ncols))
    mat.setUp()
    mat.setValuesCSR(ai, aj, av)
    mat.assemble()

    return mat
