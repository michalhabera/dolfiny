# Dolfin-y
Dolfin-y, high-level wrappers for dolfin-x, the FEniCS library.
This is an experimental version **not meant for production use.**

Dolfin-x involves many low-level operations - from handling ghosted values to interfacing PETSc solvers directly. This library serves as a wrapper around the low-level functionality of dolfin-x and is meant to bring more "old FEniCS (2019.1.0)" user API with performance and functionality of dolfin-x.

This library is written exclusively in Python.

# Installation
`pip3 install .`

# Documentation
In preparation. In the meantime please check available [demos](demo/) or [unit tests](test/).

# Authors
- Michal Habera, <michal.habera@uni.lu>,
- Andreas Zilian, <andreas.zilian@uni.lu>.

# TODO
- (4/5) add more examples for restriction functionality
  (Bingham with LM, lid-driven stationary Navier-Stokes),
- (4/5) AZ: to add few hyperelasticity examples, possibly velocity-based,
- (2) SLEPc interface, input similar to SNESBlockProblem, nonlinear f(u, lambda),
- (3) higher order MeshTags on dim < tdim,
- (1) higher-level interface for SC,
- gmsh_to_dolfin, return readable tags,
- ufl Measures addressed with readable tags

# License
Dolfin-y is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Dolfin-y is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with Dolfin-y. If not, see <http://www.gnu.org/licenses/>.