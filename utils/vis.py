from typing import Union

import pyvista as pv
import numpy as np
from trimesh import Trimesh


def visualize_mesh(mesh: Union[Trimesh, np.ndarray]):
    """
    Visualize the mesh using pyvista.
    :param mesh: input mesh or point cloud
    """
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="tan", smooth_shading=True)
    plotter.show(auto_close=False)
