import logging
import os

import numpy

from mpi4py import MPI

# def gmsh_to_dolfin(gmsh_model, tdim: int, comm=MPI.comm_world,
#                    ghost_mode=cpp.mesh.GhostMode.none, prune_y=False, prune_z=False):
#     """Converts a gmsh model object into `dolfinx.Mesh` and `dolfinx.MeshValueCollection`
#     for physical tags.

#     Parameters
#     ----------
#     gmsh_model
#     tdim
#         Topological dimension on the mesh
#     order: optional
#         Order of mesh geometry, e.g. 2 for quadratic elements.
#     comm: optional
#     ghost_mode: optional
#     prune_y: optional
#         Prune y-components. Used to embed a flat geometries into lower dimension.
#     prune_z: optional
#         Prune z-components. Used to embed a flat geometries into lower dimension.

#     Note
#     ----
#     User must call `geo.synchronize()` and `mesh.generate()` before passing the model into
#     this method.
#     """

#     logger = logging.getLogger("dolfiny")

#     # Map from internal gmsh cell type number to gmsh cell name
#     gmsh_name = {1: 'line', 2: 'triangle', 3: "quad", 5: "hexahedron",
#                  4: 'tetra', 8: 'line3', 9: 'triangle6', 10: "quad9", 11: 'tetra10',
#                  15: 'vertex'}

#     gmsh_dolfin = {"vertex": (CellType.point, 0), "line": (CellType.interval, 1),
#                    "line3": (CellType.interval, 2), "triangle": (CellType.triangle, 1),
#                    "triangle6": (CellType.triangle, 2), "quad": (CellType.quadrilateral, 1),
#                    "quad9": (CellType.quadrilateral, 2), "tetra": (CellType.tetrahedron, 1),
#                    "tetra10": (CellType.tetrahedron, 2), "hexahedron": (CellType.hexahedron, 1),
#                    "hexahedron27": (CellType.hexahedron, 2)}

#     # Number of nodes for gmsh cell type
#     nodes = {'line': 2, 'triangle': 3, 'tetra': 4, 'line3': 3,
#              'triangle6': 6, 'tetra10': 10, 'vertex': 1, "quad": 4, "quad9": 9}

#     node_tags, coord, param_coords = gmsh_model.mesh.getNodes()

#     # Fetch elements for the mesh
#     cell_types, cell_tags, cell_node_tags = gmsh_model.mesh.getElements(dim=tdim)

#     unused_nodes = numpy.setdiff1d(node_tags, cell_node_tags)
#     unused_nodes_indices = numpy.where(node_tags == unused_nodes)[0]

#     # Every node has 3 components in gmsh
#     dim = 3
#     points = numpy.reshape(coord, (-1, dim))

#     # Delete unreferenced nodes
#     points = numpy.delete(points, unused_nodes_indices, axis=0)
#     node_tags = numpy.delete(node_tags, unused_nodes_indices)

#     # Prepare a map from node tag to index in coords array
#     nmap = numpy.argsort(node_tags - 1)
#     cells = {}

#     if len(cell_types) > 1:
#         raise RuntimeError("Mixed topology meshes not supported.")

#     name = gmsh_name[cell_types[0]]
#     num_nodes = nodes[name]

#     logger.info("Processing mesh of gmsh cell name \"{}\"".format(name))

#     # Shift 1-based numbering and apply node map
#     cells[name] = nmap[cell_node_tags[0] - 1]
#     cells[name] = numpy.reshape(cells[name], (-1, num_nodes))

#     if prune_z:
#         if not numpy.allclose(points[:, 2], 0.0):
#             raise RuntimeError("Non-zero z-component would be pruned.")

#         points = points[:, :-1]

#     if prune_y:
#         if not numpy.allclose(points[:, 1], 0.0):
#             raise RuntimeError("Non-zero y-component would be pruned.")

#         if prune_z:
#             # In the case we already pruned z-component
#             points = points[:, 0]
#         else:
#             points = points[:, [0, 2]]

#     dolfin_cell_type, order = gmsh_dolfin[name]

#     permutation = cpp.io.permutation_vtk_to_dolfin(dolfin_cell_type, num_nodes)
#     logger.info("Mesh will be permuted with {}".format(permutation))
#     cells[name][:, :] = cells[name][:, permutation]

#     logger.info("Constructing mesh for tdim: {}, gdim: {}".format(tdim, points.shape[1]))
#     logger.info("Number of elements: {}".format(cells[name].shape[0]))

#     mesh = cpp.mesh.Mesh(comm, dolfin_cell_type, points,
#                          cells[name], [], ghost_mode)

#     mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)

#     mvcs = {}

#     # Get physical groups (dimension, tag)
#     pgdim_pgtags = gmsh_model.getPhysicalGroups()
#     for pgdim, pgtag in pgdim_pgtags:

#         if order > 1 and pgdim != tdim:
#             raise RuntimeError("Submanifolds for higher order mesh not supported.")

#         # For the current physical tag there could be multiple entities
#         # e.g. user tagged bottom and up boundary part with one physical tag
#         entity_tags = gmsh_model.getEntitiesForPhysicalGroup(pgdim, pgtag)

#         _mvc_cells = []
#         _mvc_data = []

#         for i, entity_tag in enumerate(entity_tags):
#             pgcell_types, pgcell_tags, pgnode_tags = gmsh_model.mesh.getElements(pgdim, entity_tag)

#             assert(len(pgcell_types) == 1)
#             pgname = gmsh_name[pgcell_types[0]]
#             pgnum_nodes = nodes[pgname]

#             # Shift 1-based numbering and apply node map
#             pgnode_tags[0] = nmap[pgnode_tags[0] - 1]
#             _mvc_cells.append(pgnode_tags[0].reshape(-1, pgnum_nodes))
#             _mvc_data.append(numpy.full(_mvc_cells[-1].shape[0], pgtag))

#         # Stack all topology and value data. This prepares data
#         # for one MVC per (dim, physical tag) instead of multiple MVCs
#         _mvc_data = numpy.hstack(_mvc_data)
#         _mvc_cells = numpy.vstack(_mvc_cells)

#         # Fetch the permutation needed for physical group
#         pgdolfin_cell_type, pgorder = gmsh_dolfin[pgname]
#         pgpermutation = cpp.io.permutation_vtk_to_dolfin(pgdolfin_cell_type, _mvc_cells.shape[1])

#         _mvc_cells[:, :] = _mvc_cells[:, pgpermutation]

#         logger.info("Constructing MVC for tdim: {}".format(pgdim))
#         logger.info("Number of data values: {}".format(_mvc_data.shape[0]))

#         mvc = MeshValueCollection("size_t", mesh, pgdim, _mvc_cells, _mvc_data)
#         mvcs[(pgdim, pgtag)] = mvc

#     return mesh, mvcs


def msh_to_xdmf(msh_file, tdim, gdim=3, prune=False, xdmf_file=None, merge_xdmf=True, comm=MPI.COMM_WORLD):
    """Converts msh file to a set of codimensionX xdmf/h5 files for use in dolfinx.

    Parameters
    ----------
    msh_file
        Name of .msh file (incl. extension)
    tdim
        Topological dimension of the mesh
    gdim: optional
        Geometrical dimension of the mesh
    prune: optional
        Prune z-components from points geometry, i.e. embed the mesh into XY plane
    xdmf_file: optional
        XDMF file for output, subdomains/interfaces extension is derived from base name
    comm: optional
        MPI communicator

    Returns
    -------
    The generated xdmf files as strings.

    """

    logger = logging.getLogger("dolfiny")

    if xdmf_file is None:
        path = os.path.dirname(os.path.abspath(msh_file))
        base = os.path.splitext(os.path.basename(msh_file))[0]
    else:
        path = os.path.dirname(os.path.abspath(xdmf_file))
        base = os.path.splitext(os.path.basename(xdmf_file))[0]

    xdmf_codimension = {}

    label_meshtag_map = None

    # Conversion with meshio is serial
    if comm.rank == 0:

        import meshio

        logger.info(f"Reading Gmsh mesh into meshio from path {msh_file:s}")
        mesh = meshio.read(msh_file)

        if prune:
            mesh.prune()

        points_pruned = mesh.points[:, :gdim]  # set active coordinate components

        table_cell_types = {  # meshio cell types per topological dimension
            3: ["tetra", "hexahedron", "tetra10", "hexahedron20"],
            2: ["triangle", "quad", "triangle6", "quad8"],
            1: ["line", "line3"],
            0: ["vertex"]}

        # Extract relevant entity blocks depending on supported cell types
        for codim in range(0, tdim + 1):

            cell_types = list(set([cb.type for cb in mesh.cells if cb.type in table_cell_types[tdim - codim]]))

            if len(cell_types) > 1:
                raise RuntimeError("Mixed topology meshes not supported.")

            if len(cell_types) == 1:

                entity = cell_types[0]

                entity_dolfin_supported = [(entity, mesh.get_cells_type(entity))]

                # Gmsh may invert the entity orientation and flip the sign of the tag,
                # which is reverted with abs(). This way chosen tags are kept consistent.
                celldata_entity_dolfin_supported = \
                    {"codimension{codim:1d}": [numpy.uint64(abs(mesh.get_cell_data("gmsh:physical", entity)))]}

                logger.info("Writing codimension {codim:1d} data for dolfin MeshTags")

                xdmf_codimension[codim] = f"{path:s}/{base:s}_codimension{codim:1d}.xdmf"

                celldata_mesh_dolfin_supported = meshio.Mesh(points=points_pruned,
                                                             cells=entity_dolfin_supported,
                                                             cell_data=celldata_entity_dolfin_supported)

                meshio.write(xdmf_codimension[codim], celldata_mesh_dolfin_supported)

        # Extract relevant field data for supported cell blocks
        label_meshtag_map = mesh.field_data

    # Broadcast mapping: label -> meshtag
    label_meshtag_map = comm.bcast(label_meshtag_map, root=0)

    # Broadcast xdmf_codimension files
    xdmf_codimension = comm.bcast(xdmf_codimension, root=0)

    # Optional: Merge into single xdfm/h5
    if merge_xdmf:

        # Produce single xdmf/h5 file
        xdmfs_to_xdmf(xdmf_codimension, xdmf_file)

        # Remove subdomains and interfaces files
        objs = xdmf_codimension.values()
        exts = ["xdmf", "h5"]
        if comm.rank == 0:
            for f in [f"{os.path.splitext(obj)[0]:s}.{ext:s}" for obj in objs for ext in exts]:
                try:
                    os.remove(f)
                except OSError:
                    raise Exception("Cannot remove file '%s'." % f)

        assert os.path.exists(xdmf_file), "Mesh generation as xdfm/h5 file failed!"

    comm.barrier()

    return label_meshtag_map


def xdmfs_to_xdmf(xdmf_codimension, xdmf_file, comm=MPI.COMM_WORLD):
    """Merges a set of codimensionX xdmf/h5 files into a single xdmf/h5 file.

    Parameters
    ----------
    xdmf_codimension
        Dictionary of xdmf files containing the codimension mesh data
    xdmf_file
        Name of the single xdmf file containing the merged data
    comm: optional
        MPI communicator

    Returns
    -------
    The generated xdmf file as string.


    """

    logger = logging.getLogger("dolfiny")

    path = os.path.dirname(os.path.abspath(xdmf_file))
    base = os.path.splitext(os.path.basename(xdmf_file))[0]

    xdmf_file = f"{path:s}/{base:s}.xdmf"

    # Combine meshio outputs into one mesh using dolfin
    from dolfinx.io import XDMFFile

    meshtags = {}

    for codim in sorted(xdmf_codimension):
        logger.info(f"Reading XDMF codimension from {xdmf_codimension[codim]:s}")
        with XDMFFile(comm, xdmf_codimension[codim], "r") as ifile:
            if codim == 0:
                mesh = ifile.read_mesh(name="Grid")
                mesh.topology.create_connectivity_all()
            meshtags[codim] = ifile.read_meshtags(mesh, "Grid")
            meshtags[codim].name = f"codimension{codim:1d}"

    # Write mesh and meshtags to single xdmf file
    logger.info(f"Writing XDMF all-in-one file to {xdmf_file:s}")
    with XDMFFile(comm, xdmf_file, "w") as ofile:
        ofile.write_mesh(mesh)
        for codim, mt in sorted(meshtags.items()):
            ofile.write_meshtags(mt)

    return xdmf_file


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
