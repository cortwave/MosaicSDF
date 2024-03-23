import trimesh
from trimesh import Trimesh, Scene
import numpy as np
import point_cloud_utils as pcu


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
    if isinstance(mesh, Scene):
        scene = mesh
        meshes = []
        for name, val in scene.geometry.items():
            meshes.append(val)
        mesh = trimesh.util.concatenate(meshes)

    mesh = normalize_mesh(mesh)
    return mesh


def farthest_point_sampling(points, num_samples):
    """
    Simple implementation of the Farthest Point Sampling algorithm.
    points: numpy array of shape (N, 3)
    num_samples: the number of points to sample
    """
    sampled_indices = pcu.downsample_point_cloud_poisson_disk(points, radius=0.01, target_num_samples=int(num_samples * 1.2))
    replace = len(sampled_indices) < num_samples
    sampled_indices = np.random.choice(sampled_indices, num_samples, replace=replace)
    return points[sampled_indices]
