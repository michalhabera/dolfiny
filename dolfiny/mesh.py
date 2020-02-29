#!/usr/bin/env python3


def msh2xdmf(mshfile, tdim, gdim=3, prune=False):
    """
    Convert msh file to a set of [mesh, subdomains, interfaces] xdmf/h5 files for use in dolfinx.
    """

    import os

    path = os.path.dirname(os.path.abspath(mshfile))
    base = os.path.splitext(os.path.basename(mshfile))[0]

    import meshio

    print("Reading Gmsh mesh into meshio")
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
    from numpy import uint as size_t

    # extract relevant cell blocks depending on supported cell types
    subdomains_celltypes = list(set([cb.type for cb in mesh.cells if cb.type in cell_types[tdim]]))
    interfaces_celltypes = list(set([cb.type for cb in mesh.cells if cb.type in cell_types[tdim - 1]]))

    assert(len(subdomains_celltypes) <= 1)
    assert(len(interfaces_celltypes) <= 1)

    subdomains_celltype = subdomains_celltypes[0] if len(subdomains_celltypes) > 0 else None
    interfaces_celltype = interfaces_celltypes[0] if len(subdomains_celltypes) > 0 else None

    if subdomains_celltype is not None:
        subdomains_cells_dolfin_supported = [(subdomains_celltype, mesh.get_cells_type(subdomains_celltype))]
    else:
        subdomains_cells_dolfin_supported = []

    if interfaces_celltype is not None:
        interfaces_cells_dolfin_supported = [(interfaces_celltype, mesh.get_cells_type(interfaces_celltype))]
    else:
        interfaces_cells_dolfin_supported = []

    # extract relevant cell data for supported cell blocks
    if subdomains_celltype is not None:
        subdomains_celldata_dolfin_supported = {"name_to_read":
                                                [size_t(abs(mesh.get_cell_data("gmsh:physical", subdomains_celltype)))]}
    else:
        subdomains_celldata_dolfin_supported = {}

    if interfaces_celltype is not None:
        interfaces_celldata_dolfin_supported = {"name_to_read":
                                                [size_t(abs(mesh.get_cell_data("gmsh:physical", interfaces_celltype)))]}
    else:
        interfaces_celldata_dolfin_supported = {}

    print("Writing mesh for dolfin Mesh")
    meshio.write(path + "/" + base + ".xdmf", meshio.Mesh(
        points=points_pruned,
        cells=subdomains_cells_dolfin_supported
    ))

    print("Writing subdomain data for dolfin MeshValueCollection")
    meshio.write(path + "/" + base + "_subdomains" + ".xdmf", meshio.Mesh(
        points=points_pruned,
        cells=subdomains_cells_dolfin_supported,
        cell_data=subdomains_celldata_dolfin_supported
    ))

    print("Writing interface data for dolfin MeshValueCollection")
    meshio.write(path + "/" + base + "_interfaces" + ".xdmf", meshio.Mesh(
        points=points_pruned,
        cells=interfaces_cells_dolfin_supported,
        cell_data=interfaces_celldata_dolfin_supported
    ))

    return mesh


if __name__ == "__main__":
    msh2xdmf()
