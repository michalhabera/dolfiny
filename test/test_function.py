import dolfiny.function
import ufl


def test_ufl_block_extraction(V1, V2, squaremesh_5):
    mesh = squaremesh_5
    u1, v1 = ufl.TrialFunction(V1), ufl.TestFunction(V1)
    u2, v2 = ufl.TrialFunction(V2), ufl.TestFunction(V2)

    a = u1 * v1 * ufl.dx(mesh) + ufl.grad(u1 * v2)[0] * ufl.dx(mesh)

    ablocks = dolfiny.function.extract_blocks(a, [v1, v2], [u1, u2])

    assert ablocks[1][1].empty()
    assert ablocks[0][1].empty()

    L = ufl.grad(v2)[0] * ufl.dx(mesh)
    Lblocks = dolfiny.function.extract_blocks(L, [v1, v2])

    assert Lblocks[0].empty()
    assert Lblocks[1] == L
