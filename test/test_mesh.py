import numpy
import pytest

import dolfiny.mesh
import dolfinx
import dolfinx.io
import ufl

skip_in_parallel = pytest.mark.skipif(
    dolfinx.MPI.size(dolfinx.MPI.comm_world) > 1,
    reason="This test should only be run in serial.")


@skip_in_parallel
def test_simple_triangle():
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

    gmsh.model.addPhysicalGroup(1, [l0, l2], 2)
    gmsh.model.addPhysicalGroup(1, [l1], 3)

    gmsh.model.addPhysicalGroup(2, [s0], 4)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate()
    gmsh.model.mesh.refine()

    mesh, mvcs = dolfiny.mesh.gmsh_to_dolfin(gmsh.model, 2, prune_z=True)

    assert mesh.geometry.dim == 2
    assert mesh.topology.dim == 2

    mf = dolfinx.MeshFunction("size_t", mesh, mvcs[(1, 3)], 0)
    ds = ufl.Measure("ds", subdomain_data=mf, domain=mesh)

    val = dolfinx.fem.assemble_scalar(1.0 * ds(3))
    assert numpy.isclose(val, numpy.pi / 2.0, rtol=1.0e-3)
