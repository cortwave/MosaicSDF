from itertools import product

import numpy as np
from tqdm import tqdm
import torch
import trimesh
import taichi as ti
import taichi.math as tm
from pysdf import SDF
from skimage.measure import marching_cubes

from config import KERNEL_SIZE, DEVICE


def get_grid(kernel_size=KERNEL_SIZE, device=DEVICE) -> torch.Tensor:
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


def calculate_weights(X, centers, scales) -> torch.Tensor:
    """
    Calculate the weight for poibts X.
    :arg
    X: torch.Tensor, (N, 3), the points to calculate the weight.
    centers: torch.Tensor, (M, 3), the centers of the grid.
    scales: torch.Tensor, (M), the scales of the grid.
    :return
    weights: torch.Tensor, (N, M), the weights for each point and each grid.
    """
    distances = torch.abs((X.view(-1, 1, 3) - centers.view(1, -1, 3)) / scales.view(1, -1, 1))
    max_norm = torch.max(distances, dim=2).values
    weights = torch.nn.functional.relu(1 - max_norm)  # N, M
    norm_weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-5)
    return norm_weights


@ti.func
def get_corners(x: ti.float32) -> ti.Vector:
    c0 = ti.cast(ti.floor(x), ti.i32)
    c0 = tm.clamp(c0, 0, KERNEL_SIZE - 2)
    c1 = c0 + 1
    return ti.Vector([c0, c1])


@ti.func
def normalize_point(x: ti.float32) -> ti.float32:
    return (x + 1) / 2 * KERNEL_SIZE


@ti.kernel
def find_closest_vertices_indices(X: ti.types.ndarray(), indices: ti.types.ndarray(), distances: ti.types.ndarray()):
    for idx, c in X:
        if X[idx, c] < -1 or X[idx, c] > 1:
            continue
        normalized_x = normalize_point(X[idx, c])
        c0, c1 = get_corners(normalized_x)
        dist_0, dist_1 = ti.abs(normalized_x - c0), ti.abs(c1 - normalized_x)
        bounds = ti.Vector([c0, c1])
        dists = ti.Vector([dist_0, dist_1])
        for corner_idx in ti.static(range(8)):
            bound_idx = 0
            if c == 0:
                tm.cdiv
                bound_idx = (corner_idx // 4) % 2
            elif c == 1:
                bound_idx = (corner_idx // 2) % 2
            else:
                bound_idx = corner_idx % 2
            indices[idx, corner_idx, c] = bounds[bound_idx]
            distances[idx, corner_idx, c] = dists[bound_idx]


def get_values_at_indices(Vi: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Get the values from Vi at the given indices, optimized.
    :param Vi: matrix with values shape (M, k, k, k)
    :param indices: indices for Vi matrix shape (N, M, 8, 3)
    :return: values from Vi at indices shape (N, M, 8)
    """
    N, M, _, _ = indices.shape  # Extract dimensions
    m_range = torch.arange(M)[None, :, None]  # Shape: (1, M, 1) for broadcasting over M

    # Unbind the indices tensor into separate components for indexing
    I, J, K = indices.unbind(-1)  # Each will have shape (N, M, 8)

    # Use advanced indexing to gather values directly
    # This avoids the need for manual looping over the elements
    results = Vi[m_range, I, J, K]

    # results now contains the gathered values from Vi at the specified indices
    return results


def interpolate_values(values: torch.Tensor, distances: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """
    Interpolate the values at the given distances.
    :param values: the values at the corners, shape (N, M, 8)
    :param distances: the distances to the corners, shape (N, M, 8)
    :param p: the power for the interpolation
    :return: the interpolated values, shape (N, M)
    """
    distances = torch.clamp(distances, min=1e-5)
    weights = 1 / torch.pow(distances, p)
    weights = weights / weights.sum(dim=2, keepdim=True)
    return (values * weights).sum(dim=2)


def msdf_at_point(X, centers, scales, Vi) -> torch.Tensor:
    """
    Calculate the MSDF value at point X.
    :arg
    X: torch.Tensor, (N, 3), the points to calculate the weight.
    centers: torch.Tensor, (M, 3), the centers of the grid.
    scales: torch.Tensor, (M), the scales of the grid.
    Vi: torch.Tensor, (M, k, k, k), the SDF values grid.
    :return
    msdf: torch.Tensor, (N), the SDF value at the points.
    """
    weights = calculate_weights(X, centers, scales)

    centered = (X.view(-1, 1, 3) - centers.view(1, -1, 3)) / scales.view(1, -1, 1)  # N, M, 3
    centered_flat = centered.view(-1, 3).float()  # N * M, 3
    indices = torch.zeros((centered_flat.shape[0], 8, 3), dtype=torch.int32).to(DEVICE)
    distances = torch.zeros((centered_flat.shape[0], 8, 3), dtype=torch.float32).to(DEVICE)
    find_closest_vertices_indices(centered_flat, indices, distances)
    indices = indices.view(X.size(0), centers.size(0), 8, 3)
    distances = distances.view(X.size(0), centers.size(0), 8, 3)
    distances = torch.norm(distances, dim=-1)

    values_at_corners = get_values_at_indices(Vi, indices)
    values = interpolate_values(values_at_corners, distances)
    weighted_values = (values * weights).sum(dim=1)
    return weighted_values


def calculate_msdf_value(scale: float, points: torch.Tensor, mesh: trimesh.Trimesh) -> torch.Tensor:
    all_points = get_grid_points(points, scale)
    n_points, n_grids, _ = all_points.shape
    f = SDF(mesh.vertices, mesh.faces)
    all_points = all_points.detach().cpu().numpy()
    sdf_values = f(all_points.reshape(-1, 3)).reshape(n_points, n_grids)
    return torch.tensor(sdf_values, device=DEVICE)


def sample_volume(bounds, resolution, bound_delata=0.1):
    """
    Sample a 3D volume defined by 'bounds' at a given 'resolution'.
    bounds: ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    resolution: int, number of samples per dimension
    bound_delta: float, the delta to add to the bounds to avoid sampling on the boundary.
    Returns a tensor of shape (N, 3) of sampled points.
    """
    x_delta = (bounds[0][1] - bounds[0][0]) * bound_delata
    y_delta = (bounds[1][1] - bounds[1][0]) * bound_delata
    z_delta = (bounds[2][1] - bounds[2][0]) * bound_delata
    grid_x = torch.linspace(bounds[0][0] - x_delta, bounds[0][1] + x_delta, resolution)
    grid_y = torch.linspace(bounds[1][0] - y_delta, bounds[1][1] + y_delta, resolution)
    grid_z = torch.linspace(bounds[2][0] - z_delta, bounds[2][1] + z_delta, resolution)
    meshgrid = torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    points = torch.stack(meshgrid, dim=-1).reshape(-1, 3).to(DEVICE)
    return points


def get_grid_points(centers, scales, kernel_size=KERNEL_SIZE):
    grid = get_grid(kernel_size=kernel_size)
    grid_flat = grid.reshape(-1, 3).to(centers.device)
    if isinstance(scales, torch.Tensor):
        scales = scales.view(-1, 1, 1)
    all_points = centers.reshape(centers.shape[0], 1, 3) + grid_flat.view(1, -1, 3) * scales
    return all_points


INTERP_KERNEL_SIZE = 3


@ti.kernel
def sparse_sdf_to_grid(points: ti.types.ndarray(), sdf: ti.types.ndarray(), grid: ti.types.ndarray(),
                       grid_counts: ti.types.ndarray(),
                       grid_start: ti.float32, grid_end: ti.float32, grid_res: ti.int32):
    for i in range(points.shape[0]):
        x, y, z = points[i, 0], points[i, 1], points[i, 2]
        x = (x - grid_start) / (grid_end - grid_start) * grid_res
        y = (y - grid_start) / (grid_end - grid_start) * grid_res
        z = (z - grid_start) / (grid_end - grid_start) * grid_res
        cx = ti.cast(tm.floor(x), ti.i32)
        cy = ti.cast(tm.floor(y), ti.i32)
        cz = ti.cast(tm.floor(z), ti.i32)
        for rx in ti.static(range(-INTERP_KERNEL_SIZE, INTERP_KERNEL_SIZE + 1)):
            for ry in ti.static(range(-INTERP_KERNEL_SIZE, INTERP_KERNEL_SIZE + 1)):
                for rz in ti.static(range(-INTERP_KERNEL_SIZE, INTERP_KERNEL_SIZE + 1)):
                    if cx + rx >= 0 and cx + rx < grid_res and cy + ry >= 0 and cy + ry < grid_res and cz + rz >= 0 and cz + rz < grid_res:
                        grid[cx + rx, cy + ry, cz + rz] += sdf[i]
                        grid_counts[cx + rx, cy + ry, cz + rz] += 1

    for i, j, k in grid:
        if grid_counts[i, j, k] != 0:
            grid[i, j, k] /= grid_counts[i, j, k]


def reconstruct_mesh(Vi, scales, centers, resolution=256, threshold=0.0) -> trimesh.Trimesh:
    points = get_grid_points(centers, scales).view(-1, 3)
    sdf_results = Vi
    sdf_results = sdf_results.view(-1)

    grid = torch.zeros(resolution, resolution, resolution, device=DEVICE, dtype=torch.float32)
    grid_counts = torch.zeros(resolution, resolution, resolution, device=DEVICE, dtype=torch.int32)

    sparse_sdf_to_grid(points, sdf_results, grid, grid_counts, -1.1, 1.1, resolution)
    grid_sdf = grid.detach().cpu().numpy()

    grid_sdf = grid_sdf <= threshold
    verts, faces, _, _ = marching_cubes(grid_sdf, level=0.0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh
