from typing import Union

import pyvista as pv
import numpy as np
from trimesh import Trimesh


def visualize_mesh(mesh: Union[Trimesh, np.ndarray], color="tan", name: str = None):
    """
    Visualize the mesh using pyvista.
    :param mesh: input mesh or point cloud
    :param color: color of the mesh
    :param name: name of the mesh
    """
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color=color, smooth_shading=True, name=name)
    plotter.add_axes()
    plotter.show(auto_close=False)
