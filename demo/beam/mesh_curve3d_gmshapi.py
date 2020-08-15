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
            p_beg, p_end, lines = p0, p1, [l0]
        elif shape == "zline":
            p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
            p1 = gmsh.model.geo.addPoint(0.0, 0.0, L)
            l0 = gmsh.model.geo.addLine(p0, p1)
            p_beg, p_end, lines = p0, p1, [l0]
        elif shape == "slope":
            from math import pi, cos, sin
            alpha = pi / 8
            p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
            p1 = gmsh.model.geo.addPoint(cos(alpha) * L, 0.0, sin(alpha) * L)
            l0 = gmsh.model.geo.addLine(p0, p1)
            p_beg, p_end, lines = p0, p1, [l0]
        elif shape == "q_arc":
            R = 2.0 * L / gmsh.pi
            p0 = gmsh.model.geo.addPoint(0.0, 0.0, R)
            p1 = gmsh.model.geo.addPoint(R, 0.0, 0.0)
            pc = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
            l0 = gmsh.model.geo.addCircleArc(p0, pc, p1)
            p_beg, p_end, lines = p0, p1, [l0]
        elif shape == "h_arc":
            R = L / 2.0 / gmsh.pi
            p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
            p1 = gmsh.model.geo.addPoint(R, 0.0, R)
            pc = gmsh.model.geo.addPoint(0.0, 0.0, R)
            l0 = gmsh.model.geo.addCircleArc(p0, pc, p1)
            p2 = gmsh.model.geo.addPoint(0, 0.0, 2 * R)
            l1 = gmsh.model.geo.addCircleArc(p1, pc, p2)
            p_beg, p_end, lines = p0, p2, [l0, l1]
        elif shape == "f_arc":
            R = L / 2.0 / gmsh.pi
            p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
            p1 = gmsh.model.geo.addPoint(R, 0.0, R)
            pc = gmsh.model.geo.addPoint(0.0, 0.0, R)
            l0 = gmsh.model.geo.addCircleArc(p0, pc, p1)
            p2 = gmsh.model.geo.addPoint(0, 0.0, 2 * R)
            l1 = gmsh.model.geo.addCircleArc(p1, pc, p2)
            p3 = gmsh.model.geo.addPoint(-R, 0.0, R)
            l2 = gmsh.model.geo.addCircleArc(p2, pc, p3)
            p4 = gmsh.model.geo.addPoint(0, 0.0, 0.0)
            l3 = gmsh.model.geo.addCircleArc(p3, pc, p4)
            p_beg, p_end, lines = p0, p4, [l0, l1, l2, l3]
        else:
            raise RuntimeError("Unknown shape identifier \'{shape:s}\'")

        # Define physical groups for subdomains (! target tag > 0)
        domain = 1
        gmsh.model.addPhysicalGroup(tdim, lines, domain)  # all lines
        gmsh.model.setPhysicalName(tdim, domain, 'domain')

        # Determine interfaces
        # ... (not needed, we have the end points already)

        # Define physical groups for interfaces (! target tag > 0)
        beg = 1
        gmsh.model.addPhysicalGroup(tdim - 1, [p_beg], beg)
        gmsh.model.setPhysicalName(tdim - 1, beg, 'beg')
        end = 2
        gmsh.model.addPhysicalGroup(tdim - 1, [p_end], end)
        gmsh.model.setPhysicalName(tdim - 1, end, 'end')

        # Sync
        gmsh.model.geo.synchronize()

        # Set refinement along curve direction
        for line in lines:
            numNodes = int(nL / len(lines))
            gmsh.model.mesh.setTransfiniteCurve(line, numNodes=numNodes, meshType="Progression", coef=progression)

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
