from itertools import product

import numpy as np
import torch
import trimesh

from config import KERNEL_SIZE, DEVICE


def get_interpolant(kernel_size=KERNEL_SIZE, device=DEVICE) -> torch.Tensor:
    """
    Get the cubic interpolant for the given kernel size in range [-1, 1].
    """
    interpolant = torch.zeros(kernel_size, kernel_size, kernel_size, 3, device=device)
    for i, j, k in product(range(kernel_size), repeat=3):
        coord = torch.tensor([i, j, k], device=device)
        val = 2 * coord - (kernel_size - 1)
        val = val / (kernel_size - 1)
        interpolant[i, j, k, :] = val
    return interpolant


def calculate_msdf_value(scale: float, points: np.ndarray, mesh: trimesh.Trimesh) -> torch.Tensor:
    interpolant = get_interpolant().cpu().numpy()
    interpolant_flat = interpolant.reshape(-1, 3)
    all_points = points.reshape(points.shape[0], 1, 3) + interpolant_flat * scale
    n_points, n_interpolants, _ = all_points.shape
    sdf_values = trimesh.proximity.signed_distance(mesh, all_points.reshape(-1, 3)).reshape(n_points, n_interpolants)
    return torch.tensor(sdf_values, device=DEVICE)
