from mpi4py import MPI
import dolfinx
import dolfiny
import numpy
import ufl


def test_simple_triangle():

    if MPI.COMM_WORLD.rank == 0:

        import gmsh

        gmsh.initialize()
        gmsh.model.add("test")

        p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
        p1 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0)
        p2 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0)
        p3 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0)
        p4 = gmsh.model.geo.addPoint(1.0, 0.5, 0.0)

        l0 = gmsh.model.geo.addLine(p0, p1)
        l1 = gmsh.model.geo.addCircleArc(p1, p4, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p0)

        cl0 = gmsh.model.geo.addCurveLoop([l0, l1, l2, l3])
        s0 = gmsh.model.geo.addPlaneSurface([cl0])

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(1, [l0, l2], 2)
        gmsh.model.setPhysicalName(1, 2, "sides")

        gmsh.model.addPhysicalGroup(1, [l1], 3)
        gmsh.model.setPhysicalName(1, 3, "arc")

        gmsh.model.addPhysicalGroup(2, [s0], 4)
        gmsh.model.setPhysicalName(2, 4, "surface")

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate()
        gmsh.model.mesh.setOrder(2)

        gmsh_model = gmsh.model

    else:

        gmsh_model = None

    mesh, mts = dolfiny.mesh.gmsh_to_dolfin(gmsh_model, 2, prune_z=True)
    mt1, keys1 = dolfiny.mesh.merge_meshtags(mesh, mts, 1)
    mt2, keys2 = dolfiny.mesh.merge_meshtags(mesh, mts, 2)

    assert mesh.geometry.dim == 2
    assert mesh.topology.dim == 2
    assert mts["arc"].dim == 1

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as file:
        file.write_mesh(mesh)
        mesh.topology.create_connectivity(1, 2)
        file.write_meshtags(mts["arc"], mesh.geometry)

    ds = ufl.Measure("ds", subdomain_data=mt1, domain=mesh)

    form = dolfinx.fem.form(1.0 * ds(keys1["sides"]) + 1.0 * ds(keys1["arc"]))
    val = dolfinx.fem.assemble_scalar(form)

    val = mesh.comm.allreduce(val, op=MPI.SUM)
    assert numpy.isclose(val, 2.0 + 2.0 * numpy.pi * 0.5 / 2.0, rtol=1.0e-3)

    dx = ufl.Measure("dx", subdomain_data=mt2, domain=mesh)

    form = dolfinx.fem.form(1.0 * dx(keys2["surface"]))
    val = dolfinx.fem.assemble_scalar(form)

    val = mesh.comm.allreduce(val, op=MPI.SUM)
    assert numpy.isclose(val, 1.0 + numpy.pi * 0.5**2 / 2.0, rtol=1.0e-3)
