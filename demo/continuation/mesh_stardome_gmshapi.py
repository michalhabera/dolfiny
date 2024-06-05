#!/usr/bin/env python3

from mpi4py import MPI


def mesh_stardome_gmshapi(
    name="stardome",
    order=1,
    msh_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of stardome truss using the Python API of Gmsh.
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

        # Create points and lines
        points = [
            gmsh.model.geo.addPoint(+0.433, 0.25, 0.0),
            gmsh.model.geo.addPoint(0.0, 0.5, 0.0),
            gmsh.model.geo.addPoint(-0.433, 0.25, 0.0),
            gmsh.model.geo.addPoint(-0.433, -0.25, 0.0),
            gmsh.model.geo.addPoint(0.0, -0.5, 0.0),
            gmsh.model.geo.addPoint(0.433, -0.25, 0.0),
            gmsh.model.geo.addPoint(0.25, 0.0, 0.06216),
            gmsh.model.geo.addPoint(0.125, 0.2165, 0.06216),
            gmsh.model.geo.addPoint(-0.125, 0.2165, 0.06216),
            gmsh.model.geo.addPoint(-0.25, 0.0, 0.06216),
            gmsh.model.geo.addPoint(-0.125, -0.2165, 0.06216),
            gmsh.model.geo.addPoint(0.125, -0.2165, 0.06216),
            gmsh.model.geo.addPoint(0.0, 0.0, 0.08216),
            #
            gmsh.model.geo.addPoint(0.0, 0.0, 0.08216 + 0.00001),  # ext
        ]
        lines = [
            gmsh.model.geo.addLine(points[12], points[6]),
            gmsh.model.geo.addLine(points[12], points[7]),
            gmsh.model.geo.addLine(points[12], points[8]),
            gmsh.model.geo.addLine(points[12], points[9]),
            gmsh.model.geo.addLine(points[12], points[10]),
            gmsh.model.geo.addLine(points[12], points[11]),
            gmsh.model.geo.addLine(points[6], points[7]),
            gmsh.model.geo.addLine(points[7], points[8]),
            gmsh.model.geo.addLine(points[8], points[9]),
            gmsh.model.geo.addLine(points[9], points[10]),
            gmsh.model.geo.addLine(points[10], points[11]),
            gmsh.model.geo.addLine(points[11], points[6]),
            gmsh.model.geo.addLine(points[6], points[0]),
            gmsh.model.geo.addLine(points[0], points[7]),
            gmsh.model.geo.addLine(points[7], points[1]),
            gmsh.model.geo.addLine(points[1], points[8]),
            gmsh.model.geo.addLine(points[8], points[2]),
            gmsh.model.geo.addLine(points[2], points[9]),
            gmsh.model.geo.addLine(points[9], points[3]),
            gmsh.model.geo.addLine(points[3], points[10]),
            gmsh.model.geo.addLine(points[10], points[4]),
            gmsh.model.geo.addLine(points[4], points[11]),
            gmsh.model.geo.addLine(points[11], points[5]),
            gmsh.model.geo.addLine(points[5], points[6]),
            #
            gmsh.model.geo.addLine(points[12], points[13]),  # ext
        ]

        # Sync
        gmsh.model.geo.synchronize()
        # Define physical groups for subdomains (! target tag > 0)
        domain = 0
        gmsh.model.addPhysicalGroup(tdim, lines, domain)
        gmsh.model.setPhysicalName(tdim, domain, "domain")
        # Define physical groups for interfaces (! target tag > 0)
        support = 1
        gmsh.model.addPhysicalGroup(tdim - 1, points[0:6], support)
        gmsh.model.setPhysicalName(tdim - 1, support, "support")
        connect = 2
        gmsh.model.addPhysicalGroup(tdim - 1, points[6:7], connect)
        gmsh.model.setPhysicalName(tdim - 1, connect, "connect")
        verytop = 3
        gmsh.model.addPhysicalGroup(tdim - 1, points[12:13], verytop)
        gmsh.model.setPhysicalName(tdim - 1, verytop, "verytop")
        exploit = 4
        gmsh.model.addPhysicalGroup(tdim - 1, points[13:14], exploit)
        gmsh.model.setPhysicalName(tdim - 1, exploit, "exploit")

        # Set refinement along curve direction
        for line in lines:
            gmsh.model.mesh.setTransfiniteCurve(line, numNodes=2, meshType="Progression", coef=1.0)

        # Generate the mesh
        gmsh.model.mesh.generate()

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim


if __name__ == "__main__":
    mesh_stardome_gmshapi(msh_file="stardome.msh")
