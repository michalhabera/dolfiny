import logging

from mpi4py import MPI
import numpy
from dolfinx.cpp.mesh import CellType
from dolfinx import cpp
from dolfinx.mesh import create_mesh, create_meshtags, MeshTags
from dolfinx.io import ufl_mesh_from_gmsh
from dolfinx.cpp.io import distribute_entity_data


def gmsh_to_dolfin(gmsh_model, tdim: int, comm=MPI.COMM_WORLD, prune_y=False, prune_z=False):
    """Converts a gmsh model object into `dolfinx.Mesh` and `dolfinx.MeshTags`
    for physical tags.

    Parameters
    ----------
    gmsh_model
    tdim
        Topological dimension of the mesh
    comm: optional
    ghost_mode: optional
    prune_y: optional
        Prune y-components. Used to embed a flat geometries into lower dimension.
    prune_z: optional
        Prune z-components. Used to embed a flat geometries into lower dimension.

    Note
    ----
    User must call `geo.synchronize()` and `mesh.generate()` before passing the model into
    this method.
    """

    rank = comm.rank

    logger = logging.getLogger("dolfiny")

    if rank == 0:
        # Map from internal gmsh cell type number to gmsh cell name
        # see https://gitlab.onelab.info/gmsh/gmsh/blob/master/Common/GmshDefines.h#L75
        gmsh_cellname = {1: 'line',
                         2: 'triangle',
                         3: "quad",
                         4: 'tetra',
                         5: "hexahedron",
                         8: 'line3',
                         9: 'triangle6',
                         10: "quad9",
                         11: 'tetra10',
                         12: 'hexahedron27',
                         15: 'vertex',
                         21: 'triangle10',
                         26: 'line4',
                         29: 'tetra20',
                         36: 'quad16',
                         #  92: 'hexahedron64',
                         }

        gmsh_dolfin = {"vertex": (CellType.point, 0),
                       "line": (CellType.interval, 1),
                       "line3": (CellType.interval, 2),
                       "line4": (CellType.interval, 3),
                       "triangle": (CellType.triangle, 1),
                       "triangle6": (CellType.triangle, 2),
                       "triangle10": (CellType.triangle, 3),
                       "quad": (CellType.quadrilateral, 1),
                       "quad9": (CellType.quadrilateral, 2),
                       "quad16": (CellType.quadrilateral, 3),
                       "tetra": (CellType.tetrahedron, 1),
                       "tetra10": (CellType.tetrahedron, 2),
                       "tetra20": (CellType.tetrahedron, 3),
                       "hexahedron": (CellType.hexahedron, 1),
                       "hexahedron27": (CellType.hexahedron, 2),
                       #  "hexahedron64": (CellType.hexahedron, 3),
                       }

        # Number of nodes for gmsh cell type
        nodes = {'vertex': 1,
                 'line': 2,
                 'line3': 3,
                 'line4': 4,
                 'triangle': 3,
                 'triangle6': 6,
                 'triangle10': 10,
                 'tetra': 4,
                 'tetra10': 10,
                 'tetra20': 20,
                 'quad': 4,
                 'quad9': 9,
                 'quad16': 16,
                 'hexahedron': 8,
                 'hexahedron27': 27,
                 #  'hexahedron64': 64,
                 }

        # node_tags, coord, param_coords = gmsh_model.mesh.getNodes()

        # FIXME: This silences the RuntimeWarning (ctypes / PEP3118 format string) caused by Gmsh/numpy
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            node_tags, coord, param_coords = gmsh_model.mesh.getNodes()

        # Fetch elements for the mesh
        cell_types, cell_tags, cell_node_tags = gmsh_model.mesh.getElements(dim=tdim)

        unused_nodes = numpy.setdiff1d(node_tags, cell_node_tags)
        unused_nodes_indices = []

        # FIXME: This would be expensive for many unused nodes case
        for unused_node in unused_nodes:
            unused_nodes_indices.append(numpy.where(node_tags == unused_node)[0])

        unused_nodes_indices = numpy.asarray(unused_nodes_indices, dtype=numpy.uint64)

        # Every node has 3 components in gmsh
        dim = 3
        points = numpy.reshape(coord, (-1, dim))

        # Delete unreferenced nodes
        points = numpy.delete(points, unused_nodes_indices, axis=0)
        node_tags = numpy.delete(node_tags, unused_nodes_indices)

        # Prepare a map from node tag to index in coords array
        nmap = numpy.argsort(node_tags - 1)
        cells = {}

        if len(cell_types) > 1:
            raise RuntimeError("Mixed topology meshes not supported.")

        try:
            cellname = gmsh_cellname[cell_types[0]]
            celltype = cell_types[0]
        except KeyError:
            raise RuntimeError(f"Gmsh cell code {cell_types[0]:d} not supported.")

        try:
            num_nodes = nodes[cellname]
        except KeyError:
            raise RuntimeError(f"Cannot determine number of nodes for Gmsh cell type \"{cellname:s}\".")

        logger.info(f"Processing mesh of gmsh cell name \"{cellname:s}\"")

        # Shift 1-based numbering and apply node map
        cells[cellname] = nmap[cell_node_tags[0] - 1]
        cells[cellname] = numpy.reshape(cells[cellname], (-1, num_nodes))

        if prune_z:
            if not numpy.allclose(points[:, 2], 0.0):
                raise RuntimeError("Non-zero z-component would be pruned.")

            points = points[:, :-1]

        if prune_y:
            if not numpy.allclose(points[:, 1], 0.0):
                raise RuntimeError("Non-zero y-component would be pruned.")

            if prune_z:
                # In the case we already pruned z-component
                points = points[:, 0]
            else:
                points = points[:, [0, 2]]

        try:
            dolfin_cell_type, order = gmsh_dolfin[cellname]
        except KeyError:
            raise RuntimeError(f"Cannot determine dolfin cell type for Gmsh cell type \"{cellname:s}\".")

        perm = cpp.io.perm_gmsh(dolfin_cell_type, num_nodes)
        logger.info(f"Mesh will be permuted with {perm}")
        cells = cells[cellname][:, perm]

        logger.info(f"Constructing mesh for tdim: {tdim:d}, gdim: {points.shape[1]:d}")
        logger.info(f"Number of elements: {cells.shape[0]:d}")

        cells_shape, pts_shape, celltype = comm.bcast([cells.shape, points.shape, celltype], root=0)
    else:
        cells_shape, pts_shape, celltype = comm.bcast([None, None, None], root=0)
        cells = numpy.empty((0, cells_shape[1]))
        points = numpy.empty((0, pts_shape[1]))

    mesh = create_mesh(comm, cells, points, ufl_mesh_from_gmsh(celltype, pts_shape[1]))
    mts = {}

    # Get physical groups (dimension, tag)
    pgdim_pgtags = comm.bcast(gmsh_model.getPhysicalGroups() if rank == 0 else None, root=0)

    for pgdim, pgtag in pgdim_pgtags:

        # For the current physical tag there could be multiple entities
        # e.g. user tagged bottom and up boundary part with one physical tag
        entity_tags = comm.bcast(gmsh_model.getEntitiesForPhysicalGroup(pgdim, pgtag) if rank == 0 else None, root=0)
        pg_tag_name = comm.bcast(gmsh_model.getPhysicalName(pgdim, pgtag) if rank == 0 else None, root=0)

        if pg_tag_name == "":
            pg_tag_name = f"tag_{pgtag:d}"

        if rank == 0:

            _mt_cells = []
            _mt_values = []

            for i, entity_tag in enumerate(entity_tags):
                pgcell_types, pgcell_tags, pgnode_tags = gmsh_model.mesh.getElements(pgdim, entity_tag)

                assert(len(pgcell_types) == 1)
                pgcellname = gmsh_cellname[pgcell_types[0]]
                pgnum_nodes = nodes[pgcellname]

                # Shift 1-based numbering and apply node map
                pgnode_tags[0] = nmap[pgnode_tags[0] - 1]
                _mt_cells.append(pgnode_tags[0].reshape(-1, pgnum_nodes))
                _mt_values.append(numpy.full(_mt_cells[-1].shape[0], pgtag, dtype=numpy.int32))

            # Stack all topology and value data. This prepares data
            # for one MVC per (dim, physical tag) instead of multiple MVCs
            _mt_values = numpy.hstack(_mt_values)
            _mt_cells = numpy.vstack(_mt_cells)

            # Fetch the permutation needed for physical group
            pgdolfin_cell_type, pgorder = gmsh_dolfin[pgcellname]
            pgpermutation = cpp.io.perm_gmsh(pgdolfin_cell_type, pgnum_nodes)

            _mt_cells[:, :] = _mt_cells[:, pgpermutation]

            logger.info(f"Constructing MVC for tdim: {pgdim:d}")
            logger.info(f"Number of data values: {_mt_values.shape[0]:d}")

            mt_cells_shape, pgdim = comm.bcast([_mt_cells.shape, pgdim], root=0)
        else:
            mt_cells_shape, pgdim = comm.bcast([None, None], root=0)
            _mt_cells = numpy.empty((0, mt_cells_shape[1]), dtype=numpy.int64)
            _mt_values = numpy.empty((0, ), dtype=numpy.int32)

        local_entities, local_values = distribute_entity_data(mesh, pgdim, _mt_cells, _mt_values)

        mesh.topology.create_connectivity(pgdim, 0)

        mt = create_meshtags(mesh, pgdim, cpp.graph.AdjacencyList_int32(local_entities), numpy.int32(local_values))
        mt.name = pg_tag_name

        mts[pg_tag_name] = mt

    return mesh, mts


def msh_to_gmsh(msh_file, order=1, comm=MPI.COMM_WORLD):
    """Read msh file with gmsh and return the gmsh model

    Parameters
    ----------
    msh_file:
        The msh file
    order: optional
        Adjust order of gmsh mesh cells
    comm: optional

    Returns
    -------
    gmsh_model:
        The gmsh model
    tdim:
        The highest topological dimension of the mesh entities

    """

    if comm.rank == 0:

        import gmsh

        gmsh.open(msh_file)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate()
        gmsh.model.mesh.setOrder(order)

    tdim = comm.bcast(max([dim for dim, _ in gmsh.model.getEntities()]) if comm.rank == 0 else None, root=0)

    return gmsh.model if comm.rank == 0 else None, tdim


def locate_dofs_topological(V, meshtags, value):
    """Identifes the degrees of freedom of a given function space associated with a given meshtags value.

    Parameters
    ----------
    V: FunctionSpace
    meshtags: MeshTags object
    value: mesh tag value

    Returns
    -------
    The system dof indices.

    """

    from dolfinx import fem
    from numpy import where

    return fem.locate_dofs_topological(
        V, meshtags.dim, meshtags.indices[where(meshtags.values == value)[0]])


def merge_meshtags(mts, dim):
    """ Merge multiple MeshTags into one.

    Parameters
    ----------
    mts:
        List of meshtags
    dim:
        Dimension of MeshTags which should be merged. Note it is
        not possible to merge MeshTags with different dimensions into one
        MeshTags object.

    """
    mts = [(mt, name) for name, mt in mts.items() if mt.dim == dim]
    if len(mts) == 0:
        raise RuntimeError(f"Cannot find MeshTags of dimension {dim}")

    indices = numpy.hstack([mt.indices for mt, name in mts])
    values = numpy.hstack([mt.values for mt, name in mts])

    keys = {}
    for mt, name in mts:
        comm = mt.mesh.comm
        # In some cases this process could receive a MeshTags which are empty
        # We need to return correct "keys" mapping on each process, so this
        # communicates the value from processes which don't have empty meshtags
        if len(mt.values) == 0:
            value = -1
        else:
            if numpy.max(mt.values) < 0:
                raise RuntimeError("Not expecting negative values for MeshTags")
            value = int(mt.values[0])
        value = comm.allreduce(value, op=MPI.MAX)

        keys[name] = value

    indices, pos = numpy.unique(indices, return_index=True)
    mt = MeshTags(mts[0][0].mesh, dim, indices, values[pos])

    return mt, keys
