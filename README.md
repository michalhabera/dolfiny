# Python wrappers for DOLFINx
`dolfiny`, high-level wrappers for [DOLFINx](https://github.com/FEniCS/dolfinx), the [FEniCS library](https://www.fenicsproject.org).
This is an experimental version **not meant for production use**.

DOLFINx involves many low-level operations - from handling ghosted values to interfacing PETSc solvers directly. The `dolfiny` library serves as a wrapper around the low-level functionality of DOLFINx and is meant to combine a user-oriented API with the performance and functionality of DOLFINx.

This library is written exclusively in Python.

# Installation
`pip3 install .`

# Documentation
In preparation. In the meantime please check available [demos](demo/) or [unit tests](test/).

# Authors
- Michal Habera, <michal.habera@uni.lu>,
- Andreas Zilian, <andreas.zilian@uni.lu>.

# TODO
- add higher-level interface for static condensation (and demo),
- add demo of monolithic fluid-structure interaction,
- add demo of stability analysis.

# License
`dolfiny` is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

`dolfiny` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with `dolfiny`. If not, see <http://www.gnu.org/licenses/>.
