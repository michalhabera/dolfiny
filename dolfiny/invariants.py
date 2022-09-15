import ufl


def invariants_principal(A):
    """Principal invariants of (real-valued) tensor A.
    https://doi.org/10.1007/978-3-7091-0174-2_3
    """
    i1 = ufl.tr(A)
    i2 = (ufl.tr(A)**2 - ufl.tr(A * A)) / 2
    i3 = ufl.det(A)
    return i1, i2, i3


def invariants_main(A):
    """Main invariants of (real-valued) tensor A.
    https://doi.org/10.1007/978-3-7091-0174-2_3
    """
    j1 = ufl.tr(A)
    j2 = ufl.tr(A * A)
    j3 = ufl.tr(A * A * A)
    return j1, j2, j3


def eigenstate_legacy(A):
    """Eigenvalues and eigenprojectors of the 3x3 (real-valued) tensor A.
    Provides the spectral decomposition A = sum_{a=0}^{2} λ_a * E_a
    with eigenvalues λ_a and their associated eigenprojectors E_a = n_a^R x n_a^L
    ordered by magnitude.
    The eigenprojectors of eigenvalues with multiplicity n are returned as 1/n-fold projector.

    Note: Tensor A must not have complex eigenvalues!
    """
    if ufl.shape(A) != (3, 3):
        raise RuntimeError(f"Tensor A of shape {ufl.shape(A)} != (3, 3) is not supported!")
    #
    eps = 1.0e-10
    #
    A = ufl.variable(A)
    #
    # --- determine eigenvalues λ0, λ1, λ2
    #
    # additively decompose: A = tr(A) / 3 * I + dev(A) = q * I + B
    q = ufl.tr(A) / 3
    B = A - q * ufl.Identity(3)
    # observe: det(λI - A) = 0  with shift  λ = q + ω --> det(ωI - B) = 0 = ω**3 - j * ω - b
    j = ufl.tr(B * B) / 2  # == -I2(B) for trace-free B, j < 0 indicates A has complex eigenvalues
    b = ufl.tr(B * B * B) / 3  # == I3(B) for trace-free B
    # solve: 0 = ω**3 - j * ω - b  by substitution  ω = p * cos(phi)
    #        0 = p**3 * cos**3(phi) - j * p * cos(phi) - b  | * 4 / p**3
    #        0 = 4 * cos**3(phi) - 3 * cos(phi) - 4 * b / p**3  | --> p := sqrt(j * 4 / 3)
    #        0 = cos(3 * phi) - 4 * b / p**3
    #        0 = cos(3 * phi) - r                  with  -1 <= r <= +1
    #    phi_k = [acos(r) + (k + 1) * 2 * pi] / 3  for  k = 0, 1, 2
    p = 2 / ufl.sqrt(3) * ufl.sqrt(j + eps ** 2)  # eps: MMM
    r = 4 * b / p ** 3
    r = ufl.Max(ufl.Min(r, +1 - eps), -1 + eps)  # eps: LMM, MMH
    phi = ufl.acos(r) / 3
    # sorted eigenvalues: λ0 <= λ1 <= λ2
    λ0 = q + p * ufl.cos(phi + 2 / 3 * ufl.pi)  # low
    λ1 = q + p * ufl.cos(phi + 4 / 3 * ufl.pi)  # middle
    λ2 = q + p * ufl.cos(phi)  # high
    #
    # --- determine eigenprojectors E0, E1, E2
    #
    E0 = ufl.diff(λ0, A).T
    E1 = ufl.diff(λ1, A).T
    E2 = ufl.diff(λ2, A).T
    #
    return [λ0, λ1, λ2], [E0, E1, E2]


def eigenstate(A):
    """Eigenvalues and eigenprojectors of the 3x3 (real-valued) tensor A.
    Provides the spectral decomposition A = sum_{a=0}^{2} λ_a * E_a
    with (ordered) eigenvalues λ_a and their associated eigenprojectors E_a = n_a^R x n_a^L.

    Note: Tensor A must not have complex eigenvalues!
    """
    if ufl.shape(A) != (3, 3):
        raise RuntimeError(f"Tensor A of shape {ufl.shape(A)} != (3, 3) is not supported!")
    #
    eps = 3.0e-16  # slightly above 2**-(53 - 1), see https://en.wikipedia.org/wiki/IEEE_754
    #
    A = ufl.variable(A)
    #
    # --- determine eigenvalues λ0, λ1, λ2
    #
    I1, I2, I3 = invariants_principal(A)
    dq = 2 * I1**3 - 9 * I1 * I2 + 27 * I3
    #
    Δx = [
        A[0, 1] * A[1, 2] * A[2, 0] - A[0, 2] * A[1, 0] * A[2, 1],
        A[0, 1]**2 * A[1, 2] - A[0, 1] * A[0, 2] * A[1, 1] + A[0, 1] * A[0, 2] * A[2, 2] - A[0, 2]**2 * A[2, 1],
        A[0, 0] * A[0, 1] * A[2, 1] - A[0, 1]**2 * A[2, 0] - A[0, 1] * A[2, 1] * A[2, 2] + A[0, 2] * A[2, 1]**2,
        A[0, 0] * A[0, 2] * A[1, 2] + A[0, 1] * A[1, 2]**2 - A[0, 2]**2 * A[1, 0] - A[0, 2] * A[1, 1] * A[1, 2],
        A[0, 0] * A[0, 1] * A[1, 2] - A[0, 1] * A[0, 2] * A[1, 0] - A[0, 1] * A[1, 2] * A[2, 2] + A[0, 2] * A[1, 2] * A[2, 1],  # noqa: E501
        A[0, 0] * A[0, 2] * A[2, 1] - A[0, 1] * A[0, 2] * A[2, 0] + A[0, 1] * A[1, 2] * A[2, 1] - A[0, 2] * A[1, 1] * A[2, 1],  # noqa: E501
        A[0, 1] * A[1, 0] * A[1, 2] - A[0, 2] * A[1, 0] * A[1, 1] + A[0, 2] * A[1, 0] * A[2, 2] - A[0, 2] * A[1, 2] * A[2, 0],  # noqa: E501
        A[0, 0]**2 * A[1, 2] - A[0, 0] * A[0, 2] * A[1, 0] - A[0, 0] * A[1, 1] * A[1, 2] - A[0, 0] * A[1, 2] * A[2, 2] + A[0, 1] * A[1, 0] * A[1, 2] + A[0, 2] * A[1, 0] * A[2, 2] + A[1, 1] * A[1, 2] * A[2, 2] - A[1, 2]**2 * A[2, 1],  # noqa: E501
        A[0, 0]**2 * A[1, 2] - A[0, 0] * A[0, 2] * A[1, 0] - A[0, 0] * A[1, 1] * A[1, 2] - A[0, 0] * A[1, 2] * A[2, 2] + A[0, 2] * A[1, 0] * A[1, 1] + A[0, 2] * A[1, 2] * A[2, 0] + A[1, 1] * A[1, 2] * A[2, 2] - A[1, 2]**2 * A[2, 1],  # noqa: E501
        A[0, 0] * A[0, 1] * A[1, 1] - A[0, 0] * A[0, 1] * A[2, 2] - A[0, 1]**2 * A[1, 0] + A[0, 1] * A[0, 2] * A[2, 0] - A[0, 1] * A[1, 1] * A[2, 2] + A[0, 1] * A[2, 2]**2 + A[0, 2] * A[1, 1] * A[2, 1] - A[0, 2] * A[2, 1] * A[2, 2],  # noqa: E501
        A[0, 0] * A[0, 1] * A[1, 1] - A[0, 0] * A[0, 1] * A[2, 2] + A[0, 0] * A[0, 2] * A[2, 1] - A[0, 1]**2 * A[1, 0] - A[0, 1] * A[1, 1] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 1] + A[0, 1] * A[2, 2]**2 - A[0, 2] * A[2, 1] * A[2, 2],  # noqa: E501
        A[0, 0] * A[0, 1] * A[1, 2] - A[0, 0] * A[0, 2] * A[1, 1] + A[0, 0] * A[0, 2] * A[2, 2] - A[0, 1] * A[1, 1] * A[1, 2] - A[0, 2]**2 * A[2, 0] + A[0, 2] * A[1, 1]**2 - A[0, 2] * A[1, 1] * A[2, 2] + A[0, 2] * A[1, 2] * A[2, 1],  # noqa: E501
        A[0, 0] * A[0, 2] * A[1, 1] - A[0, 0] * A[0, 2] * A[2, 2] - A[0, 1] * A[0, 2] * A[1, 0] + A[0, 1] * A[1, 1] * A[1, 2] - A[0, 1] * A[1, 2] * A[2, 2] + A[0, 2]**2 * A[2, 0] - A[0, 2] * A[1, 1]**2 + A[0, 2] * A[1, 1] * A[2, 2],  # noqa: E501
        A[0, 0]**2 * A[1, 1] - A[0, 0]**2 * A[2, 2] - A[0, 0] * A[0, 1] * A[1, 0] + A[0, 0] * A[0, 2] * A[2, 0] - A[0, 0] * A[1, 1]**2 + A[0, 0] * A[2, 2]**2 + A[0, 1] * A[1, 0] * A[1, 1] - A[0, 2] * A[2, 0] * A[2, 2] + A[1, 1]**2 * A[2, 2] - A[1, 1] * A[1, 2] * A[2, 1] - A[1, 1] * A[2, 2]**2 + A[1, 2] * A[2, 1] * A[2, 2]]  # noqa: E501
    Δy = [
        A[0, 2] * A[1, 0] * A[2, 1] - A[0, 1] * A[1, 2] * A[2, 0],
        A[1, 0]**2 * A[2, 1] - A[1, 0] * A[1, 1] * A[2, 0] + A[1, 0] * A[2, 0] * A[2, 2] - A[1, 2] * A[2, 0]**2,
        A[0, 0] * A[1, 0] * A[1, 2] - A[0, 2] * A[1, 0]**2 - A[1, 0] * A[1, 2] * A[2, 2] + A[1, 2]**2 * A[2, 0],
        A[0, 0] * A[2, 0] * A[2, 1] - A[0, 1] * A[2, 0]**2 + A[1, 0] * A[2, 1]**2 - A[1, 1] * A[2, 0] * A[2, 1],
        A[0, 0] * A[1, 0] * A[2, 1] - A[0, 1] * A[1, 0] * A[2, 0] - A[1, 0] * A[2, 1] * A[2, 2] + A[1, 2] * A[2, 0] * A[2, 1],  # noqa: E501
        A[0, 0] * A[1, 2] * A[2, 0] - A[0, 2] * A[1, 0] * A[2, 0] + A[1, 0] * A[1, 2] * A[2, 1] - A[1, 1] * A[1, 2] * A[2, 0],  # noqa: E501
        A[0, 1] * A[1, 0] * A[2, 1] - A[0, 1] * A[1, 1] * A[2, 0] + A[0, 1] * A[2, 0] * A[2, 2] - A[0, 2] * A[2, 0] * A[2, 1],  # noqa: E501
        A[0, 0]**2 * A[2, 1] - A[0, 0] * A[0, 1] * A[2, 0] - A[0, 0] * A[1, 1] * A[2, 1] - A[0, 0] * A[2, 1] * A[2, 2] + A[0, 1] * A[1, 0] * A[2, 1] + A[0, 1] * A[2, 0] * A[2, 2] + A[1, 1] * A[2, 1] * A[2, 2] - A[1, 2] * A[2, 1]**2,  # noqa: E501
        A[0, 0]**2 * A[2, 1] - A[0, 0] * A[0, 1] * A[2, 0] - A[0, 0] * A[1, 1] * A[2, 1] - A[0, 0] * A[2, 1] * A[2, 2] + A[0, 1] * A[1, 1] * A[2, 0] + A[0, 2] * A[2, 0] * A[2, 1] + A[1, 1] * A[2, 1] * A[2, 2] - A[1, 2] * A[2, 1]**2,  # noqa: E501
        A[0, 0] * A[1, 0] * A[1, 1] - A[0, 0] * A[1, 0] * A[2, 2] - A[0, 1] * A[1, 0]**2 + A[0, 2] * A[1, 0] * A[2, 0] - A[1, 0] * A[1, 1] * A[2, 2] + A[1, 0] * A[2, 2]**2 + A[1, 1] * A[1, 2] * A[2, 0] - A[1, 2] * A[2, 0] * A[2, 2],  # noqa: E501
        A[0, 0] * A[1, 0] * A[1, 1] - A[0, 0] * A[1, 0] * A[2, 2] + A[0, 0] * A[1, 2] * A[2, 0] - A[0, 1] * A[1, 0]**2 - A[1, 0] * A[1, 1] * A[2, 2] + A[1, 0] * A[1, 2] * A[2, 1] + A[1, 0] * A[2, 2]**2 - A[1, 2] * A[2, 0] * A[2, 2],  # noqa: E501
        A[0, 0] * A[1, 0] * A[2, 1] - A[0, 0] * A[1, 1] * A[2, 0] + A[0, 0] * A[2, 0] * A[2, 2] - A[0, 2] * A[2, 0]**2 - A[1, 0] * A[1, 1] * A[2, 1] + A[1, 1]**2 * A[2, 0] - A[1, 1] * A[2, 0] * A[2, 2] + A[1, 2] * A[2, 0] * A[2, 1],  # noqa: E501
        A[0, 0] * A[1, 1] * A[2, 0] - A[0, 0] * A[2, 0] * A[2, 2] - A[0, 1] * A[1, 0] * A[2, 0] + A[0, 2] * A[2, 0]**2 + A[1, 0] * A[1, 1] * A[2, 1] - A[1, 0] * A[2, 1] * A[2, 2] - A[1, 1]**2 * A[2, 0] + A[1, 1] * A[2, 0] * A[2, 2],  # noqa: E501
        A[0, 0]**2 * A[1, 1] - A[0, 0]**2 * A[2, 2] - A[0, 0] * A[0, 1] * A[1, 0] + A[0, 0] * A[0, 2] * A[2, 0] - A[0, 0] * A[1, 1]**2 + A[0, 0] * A[2, 2]**2 + A[0, 1] * A[1, 0] * A[1, 1] - A[0, 2] * A[2, 0] * A[2, 2] + A[1, 1]**2 * A[2, 2] - A[1, 1] * A[1, 2] * A[2, 1] - A[1, 1] * A[2, 2]**2 + A[1, 2] * A[2, 1] * A[2, 2]]  # noqa: E501
    Δd = [9, 6, 6, 6, 8, 8, 8, 2, 2, 2, 2, 2, 2, 1]
    Δ = 0
    for i in range(len(Δd)):
        Δ += Δx[i] * Δd[i] * Δy[i]

    Δxp = [A[1, 0], A[2, 0], A[2, 1], -A[0, 0] + A[1, 1], -A[0, 0] + A[2, 2], -A[1, 1] + A[2, 2]]
    Δyp = [A[0, 1], A[0, 2], A[1, 2], -A[0, 0] + A[1, 1], -A[0, 0] + A[2, 2], -A[1, 1] + A[2, 2]]
    Δdp = [6, 6, 6, 1, 1, 1]

    dp = 0
    for i in range(len(Δdp)):
        dp += 1 / 2 * Δxp[i] * Δdp[i] * Δyp[i]

    # Avoid dp = 0 and disc = 0, both are known with absolute error of ~eps**2
    # Required to avoid sqrt(0) derivatives and negative square roots
    dp += eps**2
    Δ += eps**2

    phi3 = ufl.atan_2(ufl.sqrt(27) * ufl.sqrt(Δ), dq)

    # sorted eigenvalues: λ0 <= λ1 <= λ2
    λ = [(I1 + 2 * ufl.sqrt(dp) * ufl.cos((phi3 + 2 * ufl.pi * k) / 3)) / 3 for k in range(1, 4)]
    #
    # --- determine eigenprojectors E0, E1, E2
    #
    E = [ufl.diff(λk, A).T for λk in λ]

    return λ, E
