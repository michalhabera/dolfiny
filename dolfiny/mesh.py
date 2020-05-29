import logging
import os

from mpi4py import MPI
import numpy
from dolfinx.cpp.mesh import CellType
from dolfinx import cpp
from dolfinx.mesh import create_mesh, create_meshtags, MeshTags
from dolfinx.io import ufl_mesh_from_gmsh
from dolfinx.cpp.io import extract_local_entities


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
        gmsh_cellname = {1: 'line', 2: 'triangle', 3: "quad", 5: "hexahedron",
                         4: 'tetra', 8: 'line3', 9: 'triangle6', 10: "quad9", 11: 'tetra10',
                         15: 'vertex'}

        gmsh_dolfin = {"vertex": (CellType.point, 0), "line": (CellType.interval, 1),
                       "line3": (CellType.interval, 2), "triangle": (CellType.triangle, 1),
                       "triangle6": (CellType.triangle, 2), "quad": (CellType.quadrilateral, 1),
                       "quad9": (CellType.quadrilateral, 2), "tetra": (CellType.tetrahedron, 1),
                       "tetra10": (CellType.tetrahedron, 2), "hexahedron": (CellType.hexahedron, 1),
                       "hexahedron27": (CellType.hexahedron, 2)}

        # Number of nodes for gmsh cell type
        nodes = {'line': 2, 'triangle': 3, 'tetra': 4, 'line3': 3,
                 'triangle6': 6, 'tetra10': 10, 'vertex': 1, "quad": 4, "quad9": 9}

        node_tags, coord, param_coords = gmsh_model.mesh.getNodes()

        # Fetch elements for the mesh
        cell_types, cell_tags, cell_node_tags = gmsh_model.mesh.getElements(dim=tdim)

        unused_nodes = numpy.setdiff1d(node_tags, cell_node_tags)
        unused_nodes_indices = numpy.where(node_tags == unused_nodes)[0]

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

        cellname = gmsh_cellname[cell_types[0]]
        num_nodes = nodes[cellname]

        logger.info("Processing mesh of gmsh cell name \"{}\"".format(cellname))

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

        dolfin_cell_type, order = gmsh_dolfin[cellname]

        perm = cpp.io.perm_gmsh(dolfin_cell_type, num_nodes)
        logger.info("Mesh will be permuted with {}".format(perm))
        cells = cells[cellname][:, perm]

        logger.info("Constructing mesh for tdim: {}, gdim: {}".format(tdim, points.shape[1]))
        logger.info("Number of elements: {}".format(cells.shape[0]))

        cells_shape, pts_shape, cellname = comm.bcast([cells.shape, points.shape, cellname], root=0)
    else:
        cells_shape, pts_shape, cellname = comm.bcast([None, None, None], root=0)
        cells = numpy.empty((0, cells_shape[1]))
        points = numpy.empty((0, pts_shape[1]))

    mesh = create_mesh(comm, cells, points, ufl_mesh_from_gmsh(cellname, pts_shape[1]))
    mts = {}

    # Get physical groups (dimension, tag)
    pgdim_pgtags = comm.bcast(gmsh_model.getPhysicalGroups() if rank == 0 else None, root=0)

    for pgdim, pgtag in pgdim_pgtags:

        # For the current physical tag there could be multiple entities
        # e.g. user tagged bottom and up boundary part with one physical tag
        entity_tags = comm.bcast(gmsh_model.getEntitiesForPhysicalGroup(pgdim, pgtag) if rank == 0 else None, root=0)
        pg_tag_name = comm.bcast(gmsh_model.getPhysicalName(pgdim, pgtag) if rank == 0 else None, root=0)

        if pg_tag_name == "":
            pg_tag_name = "tag_{}".format(pgtag)

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
                _mt_values.append(numpy.full(_mt_cells[-1].shape[0], pgtag))

            # Stack all topology and value data. This prepares data
            # for one MVC per (dim, physical tag) instead of multiple MVCs
            _mt_values = numpy.hstack(_mt_values)
            _mt_cells = numpy.vstack(_mt_cells)

            # Fetch the permutation needed for physical group
            pgdolfin_cell_type, pgorder = gmsh_dolfin[pgcellname]
            pgpermutation = cpp.io.perm_gmsh(pgdolfin_cell_type, pgnum_nodes)

            _mt_cells[:, :] = _mt_cells[:, pgpermutation]

            logger.info("Constructing MVC for tdim: {}".format(pgdim))
            logger.info("Number of data values: {}".format(_mt_values.shape[0]))

            mt_cells_shape, pgdim = comm.bcast([_mt_cells.shape, pgdim], root=0)
        else:
            mt_cells_shape, pgdim = comm.bcast([None, None], root=0)
            _mt_cells = numpy.empty((0, mt_cells_shape[1]))
            _mt_values = numpy.empty((0, ))

        local_entities, local_values = extract_local_entities(mesh, pgdim, _mt_cells, _mt_values)

        mesh.topology.create_connectivity(pgdim, 0)

        mt = create_meshtags(mesh, pgdim, cpp.graph.AdjacencyList_int32(local_entities), numpy.int32(local_values))
        mt.name = pg_tag_name

        mts[pg_tag_name] = mt

    return mesh, mts


def msh_to_xdmf(mshfile, tdim, gdim=3, prune=False):
    """Converts msh file to a set of [mesh, subdomains, interfaces] xdmf/h5 files for use in dolfinx.

    Parameters
    ----------
    mshfile
        Name of .msh file (incl. extension)
    tdim
        Topological dimension of the mesh
    gdim: optional
        Geometrical dimension of the mesh
    prune:
        Prune z-components from points geometry, i.e. embedd the mesh into XY plane.

    """

    logger = logging.getLogger("dolfiny")

    path = os.path.dirname(os.path.abspath(mshfile))
    base = os.path.splitext(os.path.basename(mshfile))[0]

    import meshio

    logger.info("Reading Gmsh mesh into meshio from path {}".format(mshfile))
    mesh = meshio.read(mshfile)

    if prune:
        mesh.prune()

    points_pruned = mesh.points[:, :gdim]  # set active coordinate components

    cell_types = {  # meshio cell types per topological dimension
        3: ["tetra", "hexahedron", "tetra10", "hexahedron20"],
        2: ["triangle", "quad", "triangle6", "quad8"],
        1: ["line", "line3"],
        0: ["vertex"]}

    # The target data type for dolfin MeshValueCollection is size_t
    # Furthermore, gmsh may invert the entity orientation and flip the sign of the marker,
    # which is reverted with abs(). This way chosen labels and markers are kept consistent.

    # Extract relevant cell blocks depending on supported cell types
    subdomains_celltypes = list(set([cb.type for cb in mesh.cells if cb.type in cell_types[tdim]]))
    interfaces_celltypes = list(set([cb.type for cb in mesh.cells if cb.type in cell_types[tdim - 1]]))

    assert(len(subdomains_celltypes) <= 1)
    assert(len(interfaces_celltypes) <= 1)

    subdomains_celltype = subdomains_celltypes[0] if len(subdomains_celltypes) > 0 else None
    interfaces_celltype = interfaces_celltypes[0] if len(interfaces_celltypes) > 0 else None

    if subdomains_celltype is not None:
        subdomains_cells_dolfin_supported = [(subdomains_celltype, mesh.get_cells_type(subdomains_celltype))]
    else:
        subdomains_cells_dolfin_supported = []

    logger.info("Writing mesh for dolfin Mesh")
    meshio.write(path + "/" + base + ".xdmf", meshio.Mesh(
        points=points_pruned,
        cells=subdomains_cells_dolfin_supported
    ))

    if interfaces_celltype is not None:
        interfaces_cells_dolfin_supported = [(interfaces_celltype, mesh.get_cells_type(interfaces_celltype))]
    else:
        interfaces_cells_dolfin_supported = []

    # Extract relevant cell data for supported cell blocks
    if subdomains_celltype is not None:
        subdomains_celldata_dolfin_supported = \
            {"subdomains": [numpy.uint64(abs(mesh.get_cell_data("gmsh:physical", subdomains_celltype)))]}

        logger.info("Writing subdomain data for dolfin MeshValueCollection")
        meshio.write(path + "/" + base + "_subdomains" + ".xdmf", meshio.Mesh(
            points=points_pruned,
            cells=subdomains_cells_dolfin_supported,
            cell_data=subdomains_celldata_dolfin_supported
        ))

    if interfaces_celltype is not None:
        interfaces_celldata_dolfin_supported = \
            {"interfaces": [numpy.uint64(abs(mesh.get_cell_data("gmsh:physical", interfaces_celltype)))]}

        logger.info("Writing interface data for dolfin MeshValueCollection")
        meshio.write(path + "/" + base + "_interfaces" + ".xdmf", meshio.Mesh(
            points=points_pruned,
            cells=interfaces_cells_dolfin_supported,
            cell_data=interfaces_celldata_dolfin_supported
        ))

    return mesh


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
        comm = mt.mesh.mpi_comm()
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
