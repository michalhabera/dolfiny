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


def msh_to_xdmf(msh_file, tdim, gdim=3, prune=False, xdmf_file=None, comm=MPI.COMM_WORLD):
    """Converts msh file to a set of [mesh, subdomains, interfaces] xdmf/h5 files for use in dolfinx.

    Parameters
    ----------
    msh_file
        Name of .msh file (incl. extension)
    tdim
        Topological dimension of the mesh
    gdim: optional
        Geometrical dimension of the mesh
    prune:
        Prune z-components from points geometry, i.e. embed the mesh into XY plane
    xdmf_file:
        XDMF file for output, subdomains/interfaces extension is derived from base name
    comm:
        MPI communicator

    Returns
    -------
    The strings of the generated xdmf files.

    """

    logger = logging.getLogger("dolfiny")

    if xdmf_file is None:
        path = os.path.dirname(os.path.abspath(msh_file))
        base = os.path.splitext(os.path.basename(msh_file))[0]
    else:
        path = os.path.dirname(os.path.abspath(xdmf_file))
        base = os.path.splitext(os.path.basename(xdmf_file))[0]

    xdmf_domainmesh = f"{path:s}/{base:s}.xdmf"
    xdmf_subdomains = f"{path:s}/{base:s}_subdomains.xdmf"
    xdmf_interfaces = f"{path:s}/{base:s}_interfaces.xdmf"

    # Conversion with meshio is serial
    if comm.rank == 0:

        import meshio

        logger.info(f"Reading Gmsh mesh into meshio from path {msh_file:s}")
        mesh = meshio.read(msh_file)

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
        mesh_dolfin_supported = meshio.Mesh(points=points_pruned,
                                            cells=subdomains_cells_dolfin_supported)
        meshio.write(xdmf_domainmesh, mesh_dolfin_supported)

        if interfaces_celltype is not None:
            interfaces_cells_dolfin_supported = [(interfaces_celltype, mesh.get_cells_type(interfaces_celltype))]
        else:
            interfaces_cells_dolfin_supported = []

        # Extract relevant cell data for supported cell blocks
        if subdomains_celltype is not None:
            subdomains_celldata_dolfin_supported = \
                {"subdomains": [numpy.uint64(abs(mesh.get_cell_data("gmsh:physical", subdomains_celltype)))]}

            logger.info("Writing subdomain data for dolfin MeshTags")
            subdomains_dolfin_supported = meshio.Mesh(points=points_pruned,
                                                      cells=subdomains_cells_dolfin_supported,
                                                      cell_data=subdomains_celldata_dolfin_supported)
            meshio.write(xdmf_subdomains, subdomains_dolfin_supported)

        if interfaces_celltype is not None:
            interfaces_celldata_dolfin_supported = \
                {"interfaces": [numpy.uint64(abs(mesh.get_cell_data("gmsh:physical", interfaces_celltype)))]}

            logger.info("Writing interface data for dolfin MeshTags")
            interfaces_dolfin_supported = meshio.Mesh(points=points_pruned,
                                                      cells=interfaces_cells_dolfin_supported,
                                                      cell_data=interfaces_celldata_dolfin_supported)
            meshio.write(xdmf_interfaces, interfaces_dolfin_supported)

    comm.barrier()

    return xdmf_domainmesh, xdmf_subdomains, xdmf_interfaces


def xdmfs_to_xdmf(xdmf_domainmesh, xdmf_subdomains, xdmf_interfaces, xdmf_file=None, comm=MPI.COMM_WORLD):
    """Merges a set of separate [mesh, subdomains, interfaces] xdmf/h5 files into one single xdmf/h5 file.

    Parameters
    ----------
    xdmf_domainmesh
        Name of the xdmf file containing the domain mesh data
    xdmf_subdomains
        Name of the xdmf file containing the subdomains meshtag data
    xdmf_interfaces
        Name of the xdmf file containing the interfaces meshtag data
    xdmf_file
        Name of the single xdmf file containing the merged data

    Returns
    -------
    The string of the generated xdmf file.

    """

    logger = logging.getLogger("dolfiny")

    if xdmf_file is None:
        path = os.path.dirname(os.path.abspath(xdmf_domainmesh))
        base = os.path.splitext(os.path.basename(xdmf_domainmesh))[0] + "_merged"
    else:
        path = os.path.dirname(os.path.abspath(xdmf_file))
        base = os.path.splitext(os.path.basename(xdmf_file))[0]

    xdmf_file = f"{path:s}/{base:s}.xdmf"

    # Combine meshio outputs into one mesh using dolfin
    from dolfinx.io import XDMFFile

    # Read mesh, subdomains and interfaces from separate input files
    logger.info(f"Reading XDMF domain mesh from {xdmf_domainmesh:s}")
    with XDMFFile(comm, xdmf_domainmesh, "r") as ifile:
        mesh = ifile.read_mesh("Grid")
        mesh.topology.create_connectivity_all()

    logger.info(f"Reading XDMF subdomain meshtags from {xdmf_subdomains:s}")
    with XDMFFile(comm, xdmf_subdomains, "r") as ifile:
        subdomains = ifile.read_meshtags(mesh, "Grid")

    logger.info(f"Reading XDMF interfaces meshtags from {xdmf_interfaces:s}")
    with XDMFFile(comm, xdmf_interfaces, "r") as ifile:
        interfaces = ifile.read_meshtags(mesh, "Grid")

    # Give MeshTags proper names
    subdomains.name = "Subdomains"
    interfaces.name = "Interfaces"

    # Write mesh, subdomains and interfaces to single output file
    logger.info(f"Writing XDMF all-in-one file to {xdmf_file:s}")
    with XDMFFile(comm, xdmf_file, "w") as ofile:
        ofile.write_mesh(mesh)
        ofile.write_meshtags(subdomains)
        ofile.write_meshtags(interfaces)

    return xdmf_file
