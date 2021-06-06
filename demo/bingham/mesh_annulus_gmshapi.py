#!/usr/bin/env python3

from mpi4py import MPI


def mesh_annulus_gmshapi(name="annulus", iR=0.5, oR=3.0, nR=21, nT=16, x0=0.0, y0=0.0,
                         do_quads=False, progression=1.0, order=1, msh_file=None, comm=MPI.COMM_WORLD):
    """
    Create mesh of 2d annulus using the Python API of Gmsh.
    """
    px, py, pz = x0, y0, 0.0  # center point
    ax, ay, az = 0.0, 0.0, 1.0  # rotation axis

    tdim = 2  # target topological dimension

    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:

        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        # Add model under given name
        gmsh.model.add(name)

        # Create points and line
        p0 = gmsh.model.geo.addPoint(iR + x0, 0.0 + y0, 0.0)
        p1 = gmsh.model.geo.addPoint(oR + x0, 0.0 + y0, 0.0)
        l0 = gmsh.model.geo.addLine(p0, p1)

        # Create sectors by revolving the cross-sectional line
        s0 = gmsh.model.geo.revolve([(1, l0)], px, py, pz, ax, ay, az,
                                    angle=gmsh.pi / 2, numElements=[nT], recombine=do_quads)
        s1 = gmsh.model.geo.revolve([s0[0]], px, py, pz, ax, ay, az,
                                    angle=gmsh.pi / 2, numElements=[nT], recombine=do_quads)
        s2 = gmsh.model.geo.revolve([s1[0]], px, py, pz, ax, ay, az,
                                    angle=gmsh.pi / 2, numElements=[nT], recombine=do_quads)
        s3 = gmsh.model.geo.revolve([s2[0]], px, py, pz, ax, ay, az,
                                    angle=gmsh.pi / 2, numElements=[nT], recombine=do_quads)

        # Sync
        gmsh.model.geo.synchronize()

        # Define physical groups for subdomains (! target tag > 0)
        domain = 1
        gmsh.model.addPhysicalGroup(tdim, [s0[1][1], s1[1][1], s2[1][1], s3[1][1]], domain)  # all sectors
        gmsh.model.setPhysicalName(tdim, domain, 'domain')

        # Determine boundaries
        bs0 = gmsh.model.getBoundary(s0)
        bs1 = gmsh.model.getBoundary(s1)
        bs2 = gmsh.model.getBoundary(s2)
        bs3 = gmsh.model.getBoundary(s3)

        # Define physical groups for interfaces (! target tag > 0); boundary idx 4 and idx 5 obtained by inspection
        ring_inner = 1
        gmsh.model.addPhysicalGroup(tdim - 1, [bs0[4][1], bs1[4][1], bs2[4][1], bs3[4][1]], ring_inner)
        gmsh.model.setPhysicalName(tdim - 1, ring_inner, 'ring_inner')
        ring_outer = 2
        gmsh.model.addPhysicalGroup(tdim - 1, [bs0[5][1], bs1[5][1], bs2[5][1], bs3[5][1]], ring_outer)
        gmsh.model.setPhysicalName(tdim - 1, ring_outer, 'ring_outer')

        # Sync
        gmsh.model.geo.synchronize()

        # Set refinement in radial direction
        gmsh.model.mesh.setTransfiniteCurve(l0, numNodes=nR, meshType="Progression", coef=progression)

        # Ensure union jack meshing for triangular elements
        if not do_quads:
            gmsh.model.mesh.setTransfiniteSurface(s0[1][1], arrangement="Alternate")
            gmsh.model.mesh.setTransfiniteSurface(s1[1][1], arrangement="Alternate")
            gmsh.model.mesh.setTransfiniteSurface(s2[1][1], arrangement="Alternate")
            gmsh.model.mesh.setTransfiniteSurface(s3[1][1], arrangement="Alternate")

        # Generate the mesh
        gmsh.model.mesh.generate()

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim


if __name__ == "__main__":
    mesh_annulus_gmshapi(msh_file="annulus.msh")
