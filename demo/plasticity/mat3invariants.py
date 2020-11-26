import ufl


def invariants_principal(A):
    """Principal invariants of 3x3 (real-valued) tensor A.
    https://doi.org/10.1007/978-3-7091-0174-2_3
    """
    i1 = ufl.tr(A)
    # i2 = ufl.tr(ufl.cofac(A))
    i2 = (ufl.tr(A)**2 - ufl.tr(A * A)) / 2
    i3 = ufl.det(A)
    #
    return i1, i2, i3


def invariants_main(A):
    """Main invariants of 3x3 (real-valued) tensor A.
    https://doi.org/10.1007/978-3-7091-0174-2_3
    """
    j1 = ufl.tr(A)
    j2 = ufl.tr(A * A)
    j3 = ufl.tr(A * A * A)
    #
    return j1, j2, j3


def eigenstate(A):
    """Eigenvalues and eigenprojectors of 3x3 (real-valued) tensor A.
    Provides the spectral decomposition A = sum_{a=0}^{2} λ_a * E_a
    with eigenvalues λ_a and their associated eigenprojectors (E_a = n_a x n_a)
    ordered by magnitude.
    The eigenprojector of repeated eigenvalues is returned as zero projector.

    Note: The matrix A must not have complex eigenvalues!
    """
    I, Z = ufl.Identity(3), ufl.zero((3, 3))
    eps = 1.0e-10
    #
    # --- determine eigenvalues λ0, λ1, λ2
    #
    q = ufl.tr(A) / 3
    B = A - q * I
    j = ufl.inner(B, B)  # = 2 * J2(B) >= 0
    p = ufl.sqrt(j / 6)
    b = ufl.det(B)
    d = 1 / p ** 3 * b / 2
    r = ufl.Max(ufl.Min(d, +1 - eps), -1 + eps)  # FIXME: A-dependent thresholds?
    phi = ufl.asin(r) / 3
    # identify λ-multiplicity: j = 0: MMM, r = 1: MMH, r = -1: LMM, otherwise: LMH
    is_MMM, is_MMH, is_LMM = j < eps, (r - 1)**2 < 10 * eps**2, (r + 1)**2 < 10 * eps**2
    # sorted eigenvalues: λ0 <= λ1 <= λ2, special case: A = λI (MMM)
    λ0 = ufl.conditional(is_MMM, A[0, 0], q - 2 * p * ufl.cos(phi + ufl.pi / 6))  # low
    λ2 = ufl.conditional(is_MMM, A[1, 1], q + 2 * p * ufl.cos(phi - ufl.pi / 6))  # high
    λ1 = ufl.conditional(is_MMM, A[2, 2], q - 2 * p * ufl.sin(phi))  # middle
    #
    # --- determine eigenprojectors E0, E1, E2
    #
    # prepare projectors depending on λ-multiplicity
    E0_MMM, E1_MMM, E2_MMM = I, Z, Z
    # Z0 = (A - ufl.diag(ufl.as_vector([ 0, λ1, λ2]))) / λ0
    # Z1 = (A - ufl.diag(ufl.as_vector([λ0,  0, λ2]))) / λ1
    # Z2 = (A - ufl.diag(ufl.as_vector([λ0, λ1,  0]))) / λ2
    # E0_MMM, E1_MMM, E2_MMM = Z0, Z1, Z2
    # E0_MMM, E1_MMM, E2_MMM = A - λ0 * I, Z, Z  # FIXME: needed for E-based reconstruction (why?)
    E0_MMH, E1_MMH, E2_MMH = Z, (A - λ2 * I) / (λ1 - λ2), (A - λ1 * I) / (λ2 - λ1)
    E0_LMM, E1_LMM, E2_LMM = (A - λ1 * I) / (λ0 - λ1), (A - λ0 * I) / (λ1 - λ0), Z
    E0_LMH, E1_LMH, E2_LMH = (A - λ1 * I) * (A - λ2 * I) / (λ0 - λ1) / (λ0 - λ2), \
                             (A - λ2 * I) * (A - λ0 * I) / (λ1 - λ2) / (λ1 - λ0), \
                             (A - λ0 * I) * (A - λ1 * I) / (λ2 - λ0) / (λ2 - λ1)
    # sorted projectors
    E0 = ufl.conditional(is_MMM, E0_MMM, ufl.conditional(is_MMH, E0_MMH, ufl.conditional(is_LMM, E0_LMM, E0_LMH)))
    E1 = ufl.conditional(is_MMM, E1_MMM, ufl.conditional(is_MMH, E1_MMH, ufl.conditional(is_LMM, E1_LMM, E1_LMH)))
    E2 = ufl.conditional(is_MMM, E2_MMM, ufl.conditional(is_MMH, E2_MMH, ufl.conditional(is_LMM, E2_LMM, E2_LMH)))
    #
    return [λ0, λ1, λ2], [E0, E1, E2]
