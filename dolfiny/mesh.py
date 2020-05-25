import logging
import os

from mpi4py import MPI
import numpy
from dolfinx.cpp.mesh import CellType
from dolfinx import cpp
from dolfinx.mesh import create_mesh, create_meshtags, MeshTags
from dolfinx.io import ufl_mesh_from_gmsh
from dolfinx.cpp.io import extract_local_entities

from mpi4py import MPI

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


def gmsh_to_msh_to_xdmfh5(gmsh, tdim, gdim, comm=MPI.COMM_WORLD):

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:

        name = gmsh.model.getCurrent()
        
        msh_file = f"{tmp:s}/{name:s}.msh"
        
        if comm.rank == 0:
            gmsh.write(msh_file)
        
        xdmf_file_name = f"{name:s}.xdmf"
        
        msh_to_xdmf(msh_file, tdim, gdim, xdmf_file=xdmf_file_name)

    return xdmf_file_name


def msh_to_xdmf(msh_file, tdim, gdim=3, prune=False, xdmf_file=None, comm=MPI.COMM_WORLD):
    """Converts msh file to a set of codimensionX xdmf/h5 files. Uses meshio.

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
        Name of the XDMF file for merged output of mesh and codimension data
    comm: optional
        MPI communicator

    Returns
    -------
    The generated xdmf files as strings.

    """

    logger = logging.getLogger("dolfiny")

    path = os.path.dirname(os.path.abspath(msh_file))
    base = os.path.splitext(os.path.basename(msh_file))[0]

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

        # Extract mesh field data (gmsh names and tags)
        names_to_tags = mesh.field_data

        # Extract relevant entity blocks depending on supported cell types
        for codim in range(0, tdim + 1):

            cell_types = list(set([cb.type for cb in mesh.cells if cb.type in table_cell_types[tdim - codim]]))

            if len(cell_types) > 1:
                raise RuntimeError("Mixed topology meshes not supported.")

            if len(cell_types) == 1:

                entity = cell_types[0]

                entity_dolfin_supported = [(entity, mesh.get_cells_type(entity))]

                celldata_name = f"codimension{codim:1d}"

                # Gmsh may invert the entity orientation and flip the sign of the tag,
                # which is reverted with abs(). This way chosen tags are kept consistent.
                celldata_entity_dolfin_supported = \
                    {celldata_name: [numpy.uint64(abs(mesh.get_cell_data("gmsh:physical", entity)))]}

                names_to_tags_entity = { k:v[0] for k,v in names_to_tags.items() if v[1] == (tdim - codim) }

                logger.info(f"Writing codimension {codim:1d} data for dolfin MeshTags")

                xdmf_codimension[codim] = f"{path:s}/{base:s}_{celldata_name:s}.xdmf"

                celldata_mesh_dolfin_supported = meshio.Mesh(points=points_pruned,
                                                             cells=entity_dolfin_supported,
                                                             cell_data=celldata_entity_dolfin_supported)

                meshio.write(xdmf_codimension[codim], celldata_mesh_dolfin_supported)

                # Add information to XDMF: names_to_tags_entity as str(dict)
                import xml.etree.ElementTree as ET

                parser = ET.XMLParser()
                tree = ET.parse(xdmf_codimension[codim], parser)

                try:
                    mt_xml_node = tree.getroot().find(f"./Domain/Grid/Attribute[@Name='{celldata_name}']")
                    mt_xml_node.set('Information', str(names_to_tags_entity))
                    tree.write(xdmf_codimension[codim])
                except RuntimeError:
                    print(f"Attribute node with Name='{celldata_name}' not found in {xdmf_codimension[codim]}.")

    # Broadcast xdmf_codimension files
    xdmf_codimension = comm.bcast(xdmf_codimension, root=0)

    # Optional: Produce single xdmf/h5 file
    if xdmf_file is not None: xdmfs_to_xdmf(xdmf_codimension, xdmf_file)


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

    from dolfinx.io import XDMFFile
    import ast

    meshtags = {}
    labelmap = {}

    # Read mesh and meshtags from separate (meshio-generated) xdmf files
    for codim in sorted(xdmf_codimension):
        logger.info(f"Reading XDMF codimension from {xdmf_codimension[codim]:s}")
        with XDMFFile(comm, xdmf_codimension[codim], "r") as ifile:
            if codim == 0:
                mesh = ifile.read_mesh(name="Grid")
                mesh.topology.create_connectivity_all()
            meshtags[codim] = ifile.read_meshtags(mesh, "Grid")
            meshtags[codim].name = f"codimension{codim:1d}"
            # TODO: labelmap[codim] = ast.literal_eval(ifile.read_information("Grid"))

    # -- 8< --- FIXME: until read_information is available --------------------
    for codim in sorted(xdmf_codimension):
        import xml.etree.ElementTree as ET
        parser = ET.XMLParser()
        tree = ET.parse(xdmf_codimension[codim], parser)
        try:
            mt_xml_node = tree.getroot().find(f"./Domain/Grid/Attribute[@Name='{meshtags[codim].name}']")
            information = mt_xml_node.get('Information')
            labelmap[codim] = ast.literal_eval(information)
        except:
            pass
    # -- >8 -------------------------------------------------------------------

    # Write mesh and meshtags to single xdmf file
    logger.info(f"Writing XDMF all-in-one file to {xdmf_file:s}")
    with XDMFFile(comm, xdmf_file, "w") as ofile:
        ofile.write_mesh(mesh)
        for codim, mt in sorted(meshtags.items()):
            ofile.write_meshtags(mt)
            # TODO: ofile.write_information(str(labelmap[codim]), mt.name)

    # -- 8< --- FIXME: until write_information is available -------------------
    import xml.etree.ElementTree as ET
    parser = ET.XMLParser()
    tree = ET.parse(xdmf_file, parser)
    for codim, mt in sorted(meshtags.items()):
        try:
            mt_xml_node = tree.getroot().find(f"./Domain/Grid/Attribute[@Name='{mt.name}']")
            mt_xml_node.set('Information', str(labelmap[codim]))
            tree.write(xdmf_file)
        except RuntimeError:
            print(f"Attribute node with Name='{celldata_name}' not found in {xdmf_codimension[codim]}.")
    # -- >8 -------------------------------------------------------------------

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
