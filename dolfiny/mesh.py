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

    print("Writing mesh for dolfin Mesh")
    meshio.write(path + "/" + base + ".xdmf", meshio.Mesh(
        points=points_pruned,
        cells={key: mesh.cells[key] for key in cell_types[tdim] if key in mesh.cells}
    ))

    # The target data type for dolfin MeshValueCollection is size_t
    # Furthermore, gmsh may invert the entity orientation and flip the sign of the marker,
    # which is reverted with abs(). This way chosen labels and markers are kept consistent.
    from numpy import uint as size_t

    print("Writing subdomain data for dolfin MeshValueCollection")
    meshio.write(path + "/" + base + "_subdomains" + ".xdmf", meshio.Mesh(
        points=points_pruned,
        cells={key: mesh.cells[key] for key in cell_types[tdim] if key in mesh.cells},
        cell_data={key: {"name_to_read": size_t(abs(mesh.cell_data[key]["gmsh:physical"]))}
                   for key in cell_types[tdim]
                   if key in mesh.cell_data and "gmsh:physical" in mesh.cell_data[key]}
    ))

    print("Writing interface data for dolfin MeshValueCollection")
    meshio.write(path + "/" + base + "_interfaces" + ".xdmf", meshio.Mesh(
        points=points_pruned,
        cells={key: mesh.cells[key] for key in cell_types[tdim - 1] if key in mesh.cells},
        cell_data={key: {"name_to_read": size_t(abs(mesh.cell_data[key]["gmsh:physical"]))}
                   for key in cell_types[tdim - 1]
                   if key in mesh.cell_data and "gmsh:physical" in mesh.cell_data[key]}
    ))

    return mesh


if __name__ == "__main__":
    msh2xdmf()
