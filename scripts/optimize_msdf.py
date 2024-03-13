import numpy as np
import torch
from fire import Fire

from utils.mesh import load_mesh, farthest_point_sampling
from utils.vis import visualize_mesh
from scipy.spatial import cKDTree

from config import NUM_GRIDS, DEVICE
from utils.msdf import calculate_msdf_value


def calculate_scale(points: np.ndarray) -> float:
    """
    Calculate the initial scale s which is enough to cover all the mesh. I think that it can be set to the half of the
    maximum distance between two neighboring points.
    :param points: the points of the mesh
    :return: the scale of the mesh
    """
    kdtree = cKDTree(points)
    max_distance = 0
    distances, _ = kdtree.query(points, k=2)
    for distance in distances:
        max_distance = max(max_distance, distance[1])
    return max_distance / 2


def optimize_msdf(input_path: str = 'assets/octopus/model.obj', output_path: str = None):
    """
    Optimize the mesh for MSDF generation.
    :param input_path: path to the input mesh
    :param output_path: path to the output msdf representation
    """
    mesh = load_mesh(input_path)

    sampled_points = farthest_point_sampling(mesh.vertices, NUM_GRIDS)
    init_scale = calculate_scale(sampled_points)
    scales = torch.ones(NUM_GRIDS, device=DEVICE) * init_scale
    initial_msdf_values = calculate_msdf_value(init_scale, sampled_points, mesh)



if __name__ == '__main__':
    Fire(optimize_msdf)
