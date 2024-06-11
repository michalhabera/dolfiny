#!/usr/bin/env python3

from mpi4py import MPI

import dolfinx

import numpy as np


def mesh_box_onboard(
    name="box",
    minx=-1.0,
    maxx=+1.0,
    e=8,
    do_quads=False,
    order=1,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of 3d box using DOLFINx onboard utils.
    """

    if do_quads:
        c = dolfinx.cpp.mesh.CellType.hexahedron
    else:
        c = dolfinx.cpp.mesh.CellType.tetrahedron

    # Generate mesh of a simple box [m]
    mesh = dolfinx.mesh.create_box(comm, [[minx] * 3, [maxx] * 3], [e] * 3, cell_type=c)

    # Generate meshtags
    mts = {}
    for d in range(mesh.geometry.dim):
        for k, kx in enumerate((minx, maxx)):
            facets = dolfinx.mesh.locate_entities(
                mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[d], kx)
            )
            mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, np.unique(facets), 2 * d + k)
            mts[f"face_x{d}={kx:+.2f}"] = mt

    # Create connectivity
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    return mesh, mts
