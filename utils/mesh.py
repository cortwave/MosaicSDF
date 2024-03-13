import trimesh
from trimesh import Trimesh
import numpy as np


def normalize_mesh(mesh: Trimesh) -> Trimesh:
    """
    Normalize the mesh to have a unit bounding box [-1, 1] in all three dimensions.
    :param mesh:
    :return: normalized mesh
    """
    min_bound, max_bound = mesh.bounds
    center = (min_bound + max_bound) / 2
    scale = max(max_bound - min_bound) / 2
    mesh.vertices -= center
    mesh.vertices /= scale
    return mesh


def load_mesh(file_path: str) -> Trimesh:
    mesh = trimesh.load(file_path, process=False)
    mesh = normalize_mesh(mesh)
    return mesh


def sample_points(mesh: Trimesh, num_points: int) -> np.ndarray:
    """
    Sample points from the mesh.
    :param mesh: input mesh
    :param num_points: number of points to sample
    :return: sampled points
    """
    return np.array(furthest_points)
