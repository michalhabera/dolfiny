#!/usr/bin/env python3

from mpi4py import MPI
import pyvista


class Xdmf3Reader(pyvista.XdmfReader):
    _vtk_module_name = "vtkIOXdmf3"
    _vtk_class_name = "vtkXdmf3Reader"


def plot_block3d_pyvista(name, xdmf_file=None, plot_file=None, options={}, comm=MPI.COMM_WORLD):

    if comm.rank > 0:
        return

    reader = Xdmf3Reader(path=f"./{name}.xdmf")
    multiblock = reader.read()

    grid = multiblock[-1]
    grid.point_data["u"] = multiblock[0].point_data["u"]
    grid.point_data["s"] = multiblock[1].point_data["s"]

    pixels = 2048
    plotter = pyvista.Plotter(off_screen=True, window_size=[pixels, pixels], image_scale=1)
    plotter.add_axes(labels_off=True)

    sargs = dict(height=0.05, width=0.8, position_x=0.1, position_y=0.90,
                 title="von Mises stress", font_family="courier", fmt="%1.2f", color="black",
                 title_font_size=pixels // 50, label_font_size=pixels // 50)

    grid_warped = grid.warp_by_vector("u", factor=1.0)

    if not grid.get_cell(0).is_linear:
        grid_warped = grid_warped.extract_surface(nonlinear_subdivision=3)

    s = plotter.add_mesh(grid_warped, scalars="s", scalar_bar_args=sargs, cmap="coolwarm",
                         specular=0.5, specular_power=20, smooth_shading=True, split_sharp_edges=True)

    s.mapper.scalar_range = [0, 2]

    plotter.add_mesh(grid_warped, style="wireframe", color="black", line_width=pixels // 500)

    plotter.zoom_camera(1.15)

    plotter.screenshot(f"{name}_solution.png", transparent_background=False)


if __name__ == "__main__":
    plot_block3d_pyvista(name="spectral_elasticity")
