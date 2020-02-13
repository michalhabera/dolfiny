#!/usr/bin/env python3


def mesh_annulus_gmshapi(name="annulus", iR=0.5, oR=3.0, nR=21, nT=16, x0=0.0, y0=0.0, do_quads=False, progression=1.0):
    """
    Create mesh of 2d annulus using the Python API of Gmsh.
    """
    px, py, pz = x0, y0, 0.0  # center point
    ax, ay, az = 0.0, 0.0, 1.0  # rotation axis

    tdim = 2  # target topological dimension
    gdim = 2  # target geometrical dimension

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

    # Define physical groups for subdomains (! target tag > 0)
    domain = 1
    gmsh.model.addPhysicalGroup(tdim, [s0[1][1], s1[1][1], s2[1][1], s3[1][1]], domain)  # all sectors
    gmsh.model.setPhysicalName(tdim, domain, 'domain')

    # Determine boundaries
    bs0 = gmsh.model.getBoundary(s0)
    bs1 = gmsh.model.getBoundary(s1)
    bs2 = gmsh.model.getBoundary(s2)
    bs3 = gmsh.model.getBoundary(s3)

    # Define physical groups for interfaces (! target tag > 0)
    ring_inner = 1
    gmsh.model.addPhysicalGroup(tdim - 1, [bs0[4][1], bs1[4][1], bs2[4][1], bs3[4][1]], ring_inner)  # idx 4: inspection
    gmsh.model.setPhysicalName(tdim - 1, ring_inner, 'ring_inner')
    ring_outer = 2
    gmsh.model.addPhysicalGroup(tdim - 1, [bs0[5][1], bs1[5][1], bs2[5][1], bs3[5][1]], ring_outer)  # idx 5: inspection
    gmsh.model.setPhysicalName(tdim - 1, ring_outer, 'ring_outer')

    # Check and store labels
    labels = {}
    for tdim in range(tdim + 1):
        print("gmsh physical groups (topological dimension = {:1d}):".format(tdim))
        for info in gmsh.model.getPhysicalGroups(tdim):
            dim = info[0]
            tag = info[1]
            gid = gmsh.model.getPhysicalName(tdim, info[1])
            print("dim = {:1d} | tag = {:3d} | physical name = {:s}".format(dim, tag, gid))
            labels[gid] = tag

    # Sync
    gmsh.model.geo.synchronize()

    # Set refinement along cross-sectional direction
    gmsh.model.mesh.setTransfiniteCurve(l0, numNodes=nR, meshType="Progression", coef=progression)

    # Ensure union jack meshing for triangular elements
    if not do_quads:
        gmsh.model.mesh.setTransfiniteSurface(s0[1][1], arrangement="Alternate")
        gmsh.model.mesh.setTransfiniteSurface(s1[1][1], arrangement="Alternate")
        gmsh.model.mesh.setTransfiniteSurface(s2[1][1], arrangement="Alternate")
        gmsh.model.mesh.setTransfiniteSurface(s3[1][1], arrangement="Alternate")

    # Generate the mesh
    gmsh.model.mesh.generate()

    # Write the mesh
    gmsh.write(name + ".msh")

    # Finalise gmsh
    gmsh.finalize()

    # --- convert msh to xdmf/h5

    import dolfiny.mesh

    dolfiny.mesh.msh2xdmf(name + ".msh", tdim=tdim, gdim=gdim)

    # --- return labels

    return labels


if __name__ == "__main__":
    mesh_annulus_gmshapi()
