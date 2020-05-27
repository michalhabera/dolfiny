#!/usr/bin/env python3

from mpi4py import MPI


def mesh_curve3d_gmshapi(name="curve3d", shape="xline", L=1.0, nL=10,
                         progression=1.0, order=1, msh_file=None, comm=MPI.COMM_WORLD):
    """
    Create mesh of 3d curve using the Python API of Gmsh.
    """

    tdim = 1  # target topological dimension

    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:

        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        # Add model under given name
        gmsh.model.add(name)

        # Create points and line
        if shape == "xline":
            p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
            p1 = gmsh.model.geo.addPoint(L, 0.0, 0.0)
            l0 = gmsh.model.geo.addLine(p0, p1)
        elif shape == "zline":
            p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
            p1 = gmsh.model.geo.addPoint(0.0, 0.0, L)
            l0 = gmsh.model.geo.addLine(p0, p1)
        elif shape == "quarc":
            R = 2.0 * L / gmsh.pi
            p0 = gmsh.model.geo.addPoint(0.0, 0.0, R)
            p1 = gmsh.model.geo.addPoint(R, 0.0, 0.0)
            pc = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
            l0 = gmsh.model.geo.addCircleArc(p0, pc, p1)
        else:
            raise RuntimeError("Unknown shape identifier \'{shape:s}\'")

        # Define physical groups for subdomains (! target tag > 0)
        domain = 1
        gmsh.model.addPhysicalGroup(tdim, [l0], domain)  # all lines
        gmsh.model.setPhysicalName(tdim, domain, 'domain')

        # Determine interfaces
        # ... (not needed, we have the end points already)

        # Define physical groups for interfaces (! target tag > 0)
        left = 1
        gmsh.model.addPhysicalGroup(tdim - 1, [p0], left)
        gmsh.model.setPhysicalName(tdim - 1, left, 'left')
        right = 2
        gmsh.model.addPhysicalGroup(tdim - 1, [p1], right)
        gmsh.model.setPhysicalName(tdim - 1, right, 'right')

        # Sync
        gmsh.model.geo.synchronize()

        # Set refinement along curve direction
        gmsh.model.mesh.setTransfiniteCurve(l0, numNodes=nL, meshType="Progression", coef=progression)

        # Generate the mesh
        gmsh.model.mesh.generate()

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim


if __name__ == "__main__":
    mesh_curve3d_gmshapi(msh_file="curve3d.msh")
