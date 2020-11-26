#! /usr/bin/env python3

import dolfinx
import dolfinx.generation
import dolfiny.expression

import ufl
import numpy as np
from mpi4py import MPI

import mat3invariants

mesh = dolfinx.generation.UnitCubeMesh(MPI.COMM_WORLD, 10, 10, 10)
dx = ufl.dx(mesh)

# Test ufl eigenstate
A_ = np.array([[3.0, 0.0, 0],[ 0.0, 3.0, 0.0], [ 0, 0.0, 3.0]])  # MMM
# A_ = np.array([[3.0, 0.0, 0.0],[ 0.0, 3.0, 0.0], [ 0.0, 0.0, 5.0]])  # MMH
A_ = np.array([[5.0, 0.0, 0.0],[ 0.0, 6.0, 0.0], [ 0.0, 0.0, 5.0]])  # LMM
# A_ = np.array([[2.0, 0.0, 0.0],[ 0.0, 1.0, 0.0], [ 0.0, 0.0, 5.0]])  # LMH
# A_ = np.array([[5.0, 2.0, 0.0],[ 2.0, 1.0, 3.0], [ 0.0, 3.0, 6.0]])  # LMH, symmetric
# A_ = np.array([[5.0, 2.0, 0.0], [2.0, 5.0, 0.0], [-3.0, 4.0, 6.0]])  # LMH, non-symmetric but real eigenvalues
# A_ = np.random.rand(3, 3)

A = ufl.as_matrix(dolfinx.Constant(mesh, A_))
[e0, e1, e2], [E0, E1, E2] = mat3invariants.eigenstate(A)

V = dolfiny.expression.assemble(1.0, dx)
A_u = dolfiny.expression.assemble(A, dx) / V
A_s = dolfiny.expression.assemble(e0 * E0 + e1 * E1 + e2 * E2, dx) / V
print(A_u)
print(A_s)
assert np.isclose(A_u, A_).all(), "Wrong matrix from UFL!"
assert np.isclose(A_s, A_).all(), "Wrong spectral decomposition!"
print(f"e0 = {dolfiny.expression.assemble(e0, dx) / V:5.3e}")
print(f"e1 = {dolfiny.expression.assemble(e1, dx) / V:5.3e}")
print(f"e2 = {dolfiny.expression.assemble(e2, dx) / V:5.3e}")
print(f"E0 = \n{dolfiny.expression.assemble(E0, dx) / V}")
print(f"E1 = \n{dolfiny.expression.assemble(E1, dx) / V}")
print(f"E2 = \n{dolfiny.expression.assemble(E2, dx) / V}")
