import dolfinx
import ufl

import numpy as np
import pytest

import dolfiny.expression
import dolfiny.invariants


@pytest.mark.parametrize(
    "A",
    [
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),  # MMM
        np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]),  # MMM
        np.array([[5.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 5.0]]),  # LMM
        np.array([[5.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),  # MMH
        np.array([[5.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]]),  # LMH
        np.array(
            [[5.0, 2.0, 0.0], [2.0, 4.0, 3.0], [0.0, 3.0, 6.0]]
        ),  # LMH, symmetric, positive eigenvalues
        np.array([[5.0, 2.0, 0.0], [2.0, 1.0, 3.0], [0.0, 3.0, 6.0]]),  # LMH, symmetric
        np.array(
            [[5.0, 2.0, 0.0], [2.0, 5.0, 0.0], [-3.0, 4.0, 6.0]]
        ),  # LMH, non-symmetric, positive real eigenvalues
        np.array(
            [[-150.0, 334, 778], [-89, 195, 464], [5, -10, -27]]
        ),  # LMH, non-symmetric, real eigenvalues
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),  # LMM, one zero eigenvalue
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),  # MMH, two zero eigenvalues
        np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),  # MMM, three zero eigenvalues
    ],
)
def test_invariants_eigenstate(A, squaremesh_5):
    dx = ufl.dx(squaremesh_5)

    A_ = ufl.as_matrix(dolfinx.fem.Constant(squaremesh_5, A))

    [e0, e1, e2], [E0, E1, E2] = dolfiny.invariants.eigenstate(A_)

    e_num = np.sort(np.linalg.eigvals(A))
    e_sym = dolfiny.expression.assemble(ufl.as_vector([e0, e1, e2]), dx)

    assert np.isclose(e_num, e_sym, atol=1e-14).all()

    As = dolfiny.expression.assemble(e0 * E0 + e1 * E1 + e2 * E2, dx)

    assert np.isclose(A, As).all()
