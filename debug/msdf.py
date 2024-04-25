from utils.msdf import get_grid_points

from utils.mesh import load_mesh
import trimesh
import logging

import numpy as np
import torch
from fire import Fire
import taichi
import trimesh
from tqdm import tqdm
import point_cloud_utils as pcu
from matplotlib import pyplot as plt

from utils.mesh import load_mesh, farthest_point_sampling
from utils.vis import visualize_mesh
from scipy.spatial import cKDTree
from pysdf import SDF
import rerun as rr

from config import NUM_GRIDS, DEVICE, KERNEL_SIZE
from utils.msdf import calculate_msdf_value, reconstruct_mesh, msdf_at_point, reconstruct_from_sdf
from utils import msdf as msdf_utils
from scripts.optimize_msdf import calculate_scale

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def values2colors(values: np.ndarray):
    """
    :param values: N
    :return: Nx3
    """
    values = values - values.min()
    values = values / values.max()
    return plt.cm.viridis(values)


def debug_sdf(input_path: str = 'assets/octopus/model.obj',
               resolution: int = 32):
    rr.init("debug_msdf", spawn=True)
    init_mesh = load_mesh(input_path)

    w_vertices, w_faces = pcu.make_mesh_watertight(init_mesh.vertices, init_mesh.faces)
    mesh = trimesh.Trimesh(vertices=w_vertices, faces=w_faces)
    sdf_gt = SDF(mesh.vertices, mesh.faces)

    volumes = msdf_utils.sample_volume((-1, 1), resolution)
    sdf_values = sdf_gt(volumes)

    labels = [f'{x:.3f}' for x in sdf_values.reshape(-1).tolist()]
    colors = values2colors(sdf_values.reshape(-1))
    print(len(labels), sdf_values.shape)
    rr.log("grid_points", rr.Points3D(volumes, labels=labels, colors=colors, radii=np.abs(sdf_values.reshape(-1)) * 0.02))


if __name__ == '__main__':
    taichi.init(arch=taichi.gpu)
    Fire(debug_sdf)
