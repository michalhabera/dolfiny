#!/usr/bin/env python3


def mesh_annulus_gmshapi(name="annulus", iR=0.5, oR=3.0, nR=21, nT=16, x0=0.0, y0=0.0, do_quads=False, progression=1.0):
    """
    Create mesh of 2d annulus using the Python API of Gmsh.
    """
    px, py, pz = x0, y0, 0.0  # center point
    ax, ay, az = 0.0, 0.0, 1.0  # rotation axis

    # --- generate geometry and mesh with gmsh

    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    gmsh.model.add(name)

    # Create points and line
    p0 = gmsh.model.geo.addPoint(iR + x0, 0.0 + y0, 0.0)
    p1 = gmsh.model.geo.addPoint(oR + x0, 0.0 + y0, 0.0)
    l0 = gmsh.model.geo.addLine(p0, p1)

    # Create sectors by revolving the cross-sectional line
    s0 = gmsh.model.geo.revolve([1, l0], px, py, pz, ax, ay, az,
                                angle=gmsh.pi / 2, numElements=[nT], recombine=do_quads)
    s1 = gmsh.model.geo.revolve(s0[0], px, py, pz, ax, ay, az, angle=gmsh.pi / 2, numElements=[nT], recombine=do_quads)
    s2 = gmsh.model.geo.revolve(s1[0], px, py, pz, ax, ay, az, angle=gmsh.pi / 2, numElements=[nT], recombine=do_quads)
    s3 = gmsh.model.geo.revolve(s2[0], px, py, pz, ax, ay, az, angle=gmsh.pi / 2, numElements=[nT], recombine=do_quads)

    # Define physical groups for domain
    domain = gmsh.model.addPhysicalGroup(2, [s0[1][1], s1[1][1], s2[1][1], s3[1][1]])  # all sectors
    gmsh.model.setPhysicalName(2, domain, 'domain')

    # Determine boundaries
    bs0 = gmsh.model.getBoundary(s0)
    bs1 = gmsh.model.getBoundary(s1)
    bs2 = gmsh.model.getBoundary(s2)
    bs3 = gmsh.model.getBoundary(s3)

    # Define physical groups for boundaries
    ring_inner = gmsh.model.addPhysicalGroup(1, [bs0[4][1], bs1[4][1], bs2[4][1], bs3[4][1]])  # index 4 by inspection
    gmsh.model.setPhysicalName(1, ring_inner, 'boundary_ring_inner')
    ring_outer = gmsh.model.addPhysicalGroup(1, [bs0[5][1], bs1[5][1], bs2[5][1], bs3[5][1]])  # index 5 by inspection
    gmsh.model.setPhysicalName(1, ring_outer, 'boundary_ring_outer')

    # Check labels
    pg_domains = gmsh.model.getPhysicalGroups(2)
    pg_domain_name = gmsh.model.getPhysicalName(2, pg_domains[0][1])
    print(pg_domain_name)
    pg_boundaries = gmsh.model.getPhysicalGroups(1)
    pg_boundary_name = gmsh.model.getPhysicalName(1, pg_boundaries[0][1])
    print(pg_boundary_name)
    pg_boundary_name = gmsh.model.getPhysicalName(1, pg_boundaries[1][1])
    print(pg_boundary_name)

    # sync
    gmsh.model.geo.synchronize()

    # set refinement along cross-sectional direction
    gmsh.model.mesh.setTransfiniteCurve(l0, numNodes=nR, meshType="Progression", coef=progression)

    # ensure union jack meshing for triangular elements
    if not do_quads:
        gmsh.model.mesh.setTransfiniteSurface(s0[1][1], arrangement="Alternate")
        gmsh.model.mesh.setTransfiniteSurface(s1[1][1], arrangement="Alternate")
        gmsh.model.mesh.setTransfiniteSurface(s2[1][1], arrangement="Alternate")
        gmsh.model.mesh.setTransfiniteSurface(s3[1][1], arrangement="Alternate")

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

    points_pruned_z = mesh.points[:, :2]  # prune z components

    print("Writing mesh for dolfin Mesh")
    meshio.write("/tmp/" + name + ".xdmf", meshio.Mesh(
        points=points_pruned_z,
        cells=[(key, mesh.cells_dict[key]) for key in ["triangle", "quad"] if key in mesh.cells_dict]
    ))

    print("Writing subdomain data for dolfin MeshValueCollection")
    meshio.write("/tmp/" + name + "_subdomains" + ".xdmf", meshio.Mesh(
        points=points_pruned_z,
        cells=[(key, mesh.cells_dict[key]) for key in ["triangle", "quad"] if key in mesh.cells_dict],
        cell_data={"gmsh:physical": [mesh.cell_data_dict["gmsh:physical"][key]
                                     for key in ["triangle", "quad"] if key in mesh.cells_dict]}
    ))

    print("Writing boundary data for dolfin MeshValueCollection")
    meshio.write("/tmp/" + name + "_boundaries" + ".xdmf", meshio.Mesh(
        points=points_pruned_z,
        cells=[(key, mesh.cells_dict[key]) for key in ["line"] if key in mesh.cells_dict],
        cell_data={"gmsh:physical": [mesh.cell_data_dict["gmsh:physical"][key]
                                     for key in ["line"] if key in mesh.cells_dict]}
    ))

    # --- test mesh with dolfin and write

    import dolfinx as df
    import dolfinx.io as dfio

    # plain mesh
    print("Reading mesh into dolfin")
    with dfio.XDMFFile(df.MPI.comm_world, "/tmp/" + name + ".xdmf") as infile:
        mesh = infile.read_mesh(df.cpp.mesh.GhostMode.none)
    infile.close()

    print("Writing mesh from dolfin")
    with dfio.XDMFFile(df.MPI.comm_world, name + ".xdmf") as outfile:
        outfile.write(mesh)

    # subdomains
    print("Reading subdomains into dolfin")
    with dfio.XDMFFile(df.MPI.comm_world, "/tmp/" + name + "_subdomains" + ".xdmf") as infile:
        msh_subdomains = mesh
        mvc_subdomains = infile.read_mvc_size_t(msh_subdomains)
    infile.close()

    mfc_subdomains = df.cpp.mesh.MeshFunctionSizet(msh_subdomains, mvc_subdomains, 11)

    print("Writing subdomains from dolfin")
    with dfio.XDMFFile(df.MPI.comm_world, name + "_subdomains" + ".xdmf") as outfile:
        outfile.write(mfc_subdomains)

    # boundaries
    print("Reading boundaries into dolfin")
    with dfio.XDMFFile(df.MPI.comm_world, "/tmp/" + name + "_boundaries" + ".xdmf") as infile:
        msh_boundaries = mesh
        mvc_boundaries = infile.read_mvc_size_t(msh_boundaries)
    infile.close()

    mfc_boundaries = df.cpp.mesh.MeshFunctionSizet(msh_boundaries, mvc_boundaries, 11)

    print("Writing boundaries from dolfin")
    with dfio.XDMFFile(df.MPI.comm_world, name + "_boundaries" + ".xdmf") as outfile:
        outfile.write(mfc_boundaries)

    return mesh


if __name__ == "__main__":
    mesh_annulus_gmshapi()
