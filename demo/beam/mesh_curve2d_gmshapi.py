#!/usr/bin/env python3


def mesh_curve2d_gmshapi(name="curve2d", shape="xline", L=1.0, nL=10, progression=1.0):
    """
    Create mesh of 2d curve using the Python API of Gmsh.
    """
    
    tdim = 1 # target topological dimension
    gdim = 3 # target geometrical dimension

    # --- generate geometry and mesh with gmsh

    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    gmsh.model.add(name)

    # create points and line
    if shape == "xline":
        p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
        p1 = gmsh.model.geo.addPoint(  L, 0.0, 0.0)
        l0 = gmsh.model.geo.addLine(p0, p1)
    elif shape == "yline":
        p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
        p1 = gmsh.model.geo.addPoint(0.0, 0.0,   L)
        l0 = gmsh.model.geo.addLine(p0, p1)
    elif shape == "quarc":
        R = 2.0*L/3.1415
        p0 = gmsh.model.geo.addPoint(0.0, 0.0,   R)
        p1 = gmsh.model.geo.addPoint(  R, 0.0, 0.0)
        pc = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
        l0 = gmsh.model.geo.addCircleArc(p0, pc, p1)
    else:
        print("Error: Unknown shape identifier '{:s}'.".format(shape))
        exit()

    # define physical groups for subdomains (! target tag > 0)
    tag_domain = 1
    gmsh.model.addPhysicalGroup(tdim, [l0], tag_domain)  # all lines
    gmsh.model.setPhysicalName(tdim, tag_domain, 'subdomain_all')

    # determine interfaces
    # ... (not needed, we have the end points already)

    # define physical groups for interfaces (! target tag > 0)
    left = 1
    gmsh.model.addPhysicalGroup(tdim-1, [p0], left)
    gmsh.model.setPhysicalName(tdim-1, left, 'interface_left')
    right = 2
    gmsh.model.addPhysicalGroup(tdim-1, [p1], right)
    gmsh.model.setPhysicalName(tdim-1, right, 'interface_right')

    # check and store labels
    labels = {}
    for tdim in range(tdim+1):
        print("gmsh physical groups (topological dimension = {:1d}):".format(tdim))
        for info in gmsh.model.getPhysicalGroups(tdim):
            dim = info[0]
            tag = info[1]
            gid = gmsh.model.getPhysicalName(tdim, info[1])
            print("dim = {:1d} | tag = {:3d} | physical name = {:s}".format(dim, tag, gid))
            labels[gid] = tag
    
    # sync
    gmsh.model.geo.synchronize()

    # set refinement along curve direction
    gmsh.model.mesh.setTransfiniteCurve(l0, numNodes=nL, meshType="Progression", coef=progression)

    # generate the mesh
    gmsh.model.mesh.generate()

    # write the mesh
    gmsh.write("/tmp/" + name + ".msh")

    gmsh.finalize()

    # --- convert mesh with meshio

    import meshio

    print("Reading Gmsh mesh into meshio")
    mesh = meshio.read("/tmp/" + name + ".msh")
    # mesh.prune()

    points_pruned = mesh.points[:, :gdim]  # set active coordinate components

    cell_types = { 3: ["tetra", "hex"], 2: ["triangle", "quad"], 1: ["line"], 0: ["vertex"] }

    print("Writing mesh for dolfin Mesh")
    meshio.write("/tmp/" + name + ".xdmf", meshio.Mesh(
        points=points_pruned,
        cells={key: mesh.cells[key] for key in cell_types[tdim] if key in mesh.cells}
    ))

    print("Writing subdomain data for dolfin MeshValueCollection")
    meshio.write("/tmp/" + name + "_subdomains" + ".xdmf", meshio.Mesh(
        points=points_pruned,
        cells={key: mesh.cells[key] for key in cell_types[tdim] if key in mesh.cells},
        cell_data={key: {"name_to_read": mesh.cell_data[key]["gmsh:physical"]}
                   for key in cell_types[tdim] 
                   if key in mesh.cell_data and "gmsh:physical" in mesh.cell_data[key]}
    ))

    print("Writing interface data for dolfin MeshValueCollection")
    meshio.write("/tmp/" + name + "_interfaces" + ".xdmf", meshio.Mesh(
        points=points_pruned,
        cells={key: mesh.cells[key] for key in cell_types[tdim-1] if key in mesh.cells},
        cell_data={key: {"name_to_read": mesh.cell_data[key]["gmsh:physical"]} 
                   for key in cell_types[tdim-1] 
                   if key in mesh.cell_data and "gmsh:physical" in mesh.cell_data[key]}
    ))

    # --- test mesh with dolfin and write

    import dolfin as df
    import dolfin.io as dfio

    # plain mesh
    print("Reading mesh into dolfin")
    with dfio.XDMFFile(df.MPI.comm_world, "/tmp/" + name + ".xdmf") as infile:
        mesh = infile.read_mesh(df.cpp.mesh.GhostMode.none)
    infile.close()

    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    print(" --- dolfin: topological dimension = {}".format(tdim))
    print(" --- dolfin: geometrical dimension = {}".format(gdim))

    print("Writing mesh from dolfin")
    with dfio.XDMFFile(df.MPI.comm_world, name + ".xdmf") as outfile:
        outfile.write(mesh)

    # subdomains
    print("Reading subdomains into dolfin")
    with dfio.XDMFFile(df.MPI.comm_world, "/tmp/" + name + "_subdomains" + ".xdmf") as infile:
        msh_subdomains = mesh
        mvc_subdomains = infile.read_mvc_size_t(msh_subdomains)
    infile.close()

    mfc_subdomains = df.cpp.mesh.MeshFunctionSizet(msh_subdomains, mvc_subdomains, 0)

    print("Writing subdomains from dolfin")
    with dfio.XDMFFile(df.MPI.comm_world, name + "_subdomains" + ".xdmf") as outfile:
        outfile.write(mfc_subdomains)

    # boundaries
    print("Reading interfaces into dolfin")
    with dfio.XDMFFile(df.MPI.comm_world, "/tmp/" + name + "_interfaces" + ".xdmf") as infile:
        msh_interfaces = mesh
        mvc_interfaces = infile.read_mvc_size_t(msh_interfaces)
    infile.close()

    mfc_interfaces = df.cpp.mesh.MeshFunctionSizet(msh_interfaces, mvc_interfaces, 0)

    print("Writing interfaces from dolfin")
    with dfio.XDMFFile(df.MPI.comm_world, name + "_interfaces" + ".xdmf") as outfile:
        outfile.write(mfc_interfaces)

    return mesh, labels


if __name__ == "__main__":
    mesh_curve2d_gmshapi()
