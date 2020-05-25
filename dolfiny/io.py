import logging


def write(xdmf_file, mesh, mts=None):
    """Write mesh and meshtags to XDMFFile

    Parameters
    ----------
    xdmf_file:
        The XDMF file
    mesh:
        The dolfin mesh
    mts: optional
        The dict of MeshTags

    """
    logger = logging.getLogger("dolfiny")

    logger.info("Writing mesh")
    xdmf_file.write_mesh(mesh)

    if mts is None:
        return

    keys_meshtags = {key: mt.dim for key, mt in mts.items()}

    logger.info("Writing tag keys")
    xdmf_file.write_information("KeysOfMeshTags", str(keys_meshtags))

    logger.info("Writing meshtags")
    for mt in mts.values():
        mesh.topology.create_connectivity(mt.dim, mesh.topology.dim)
        xdmf_file.write_meshtags(mt)


def read(xdmf_file):
    """Read mesh and meshtags from XDMFFile

    Parameters
    ----------
    xdmf_file:
        The msh file

    Returns
    -------
    mesh:
        The dolfin mesh
    mts:
        The dict of meshtags

    """

    logger = logging.getLogger("dolfiny")

    logger.info("Reading mesh")
    mesh = xdmf_file.read_mesh(name="mesh")

    mesh.topology.create_connectivity_all()

    logger.info("Reading tag keys")
    value = xdmf_file.read_information("KeysOfMeshTags")

    import ast
    keys_meshtags = ast.literal_eval(value)

    mts = {}

    logger.info("Reading meshtags")
    for key in keys_meshtags.keys():
        mts[key] = xdmf_file.read_meshtags(mesh, key)

    return mesh, mts
