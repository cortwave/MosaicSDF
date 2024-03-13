import trimesh
from trimesh import Trimesh
import numpy as np
import open3d as o3d


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


def farthest_point_sampling(points, num_samples):
    """
    Simple implementation of the Farthest Point Sampling algorithm.
    points: numpy array of shape (N, 3)
    num_samples: the number of points to sample
    """
    N = points.shape[0]
    distances = np.full(N, np.inf)
    sampled_indices = np.zeros(num_samples, dtype=int)
    # Randomly choose the first point
    sampled_indices[0] = np.random.randint(N)

    for i in range(1, num_samples):
        # Update the distances based on the last added point
        dist = np.sum((points - points[sampled_indices[i - 1]]) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        # Choose the point with the maximum distance
        sampled_indices[i] = np.argmax(distances)

    return points[sampled_indices]
