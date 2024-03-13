from typing import Union

import pyvista as pv
import numpy as np
from trimesh import Trimesh


def visualize_mesh(mesh: Union[Trimesh, np.ndarray], color="tan"):
    """
    Visualize the mesh using pyvista.
    :param mesh: input mesh or point cloud
    :param point_size: size of the points
    :param color: color of the mesh
    """
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color=color, smooth_shading=True)
    plotter.show(auto_close=False)
