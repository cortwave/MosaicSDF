from itertools import product

import numpy as np
from tqdm import tqdm
import torch
import trimesh
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
    Calculate the weight for points X.
    :arg
    X: torch.Tensor, (N, 3), the points to calculate the weight.
    centers: torch.Tensor, (M, 3), the centers of the grid.
    scales: torch.Tensor, (M), the scales of the grid.
    :return
    weights: torch.Tensor, (N, M), the weights for each point and each grid.
    """
    distances = (X.view(-1, 1, 3) - centers.view(1, -1, 3)) / scales.view(1, -1, 1)
    max_norm = torch.norm(distances, p=torch.inf, dim=2)
    weights = torch.nn.functional.relu(1 - max_norm)  # N, M
    norm_weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
    return norm_weights


def find_closest_vertices_indices(x):
    # this code was taken from the hashgrid implementation
    # https://github.com/Ending2015a/hash-grid-encoding/blob/master/encoding.py#L109
    dim = x.shape[-1]
    n_neigs = 2 ** dim
    neigs = np.arange(n_neigs, dtype=np.int64).reshape((-1, 1))
    dims = np.arange(dim, dtype=np.int64).reshape((1, -1))
    bin_mask = torch.tensor(neigs & (1 << dims) == 0, dtype=bool).to(x.device)  # (neig, dim)

    bdims = len(x.shape[:-1])
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2 * (KERNEL_SIZE - 1)
    xi = x.long()
    xi = torch.clamp(xi, 0, max=KERNEL_SIZE - 2)
    xf = x - xi.float().detach()

    xi = xi.unsqueeze(dim=-2)  # (b..., 1, dim)
    xf = xf.unsqueeze(dim=-2)  # (b..., 1, dim)
    # to match the input batch shape
    bin_mask = bin_mask.reshape((1,) * bdims + bin_mask.shape)  # (1..., neig, dim)
    # get neighbors' indices and weights on each dim
    inds = torch.where(bin_mask, xi, xi + 1)  # (b..., neig, dim)
    ws = torch.where(bin_mask, 1 - xf, xf)  # (b...., neig, dim)
    # aggregate nehgibors' interp weights
    w = ws.prod(dim=-1, keepdim=True)  # (b..., neig, 1)
    return w, inds  # (b..., feat)


def get_values_at_indices(Vi: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Get the values from Vi at the given indices, optimized.
    :param Vi: matrix with values shape (M, k, k, k)
    :param indices: indices for Vi matrix shape (N, M, 8, 3)
    :return: values from Vi at indices shape (N, M, 8)
    """

    N, M, _, _ = indices.shape  # Extract dimensions
    m_range = torch.arange(M)[None, :, None].to(Vi.device)  # Shape: (1, M, 1) for broadcasting over M
    I, J, K = indices.to(Vi.device).unbind(-1)  # Each will have shape (N, M, 8)
    results = Vi[m_range, I, J, K]
    return results


def msdf_at_point_batched(X, centers, scales, Vi, batch_size: int = 4096 * 16) -> torch.Tensor:
    result = []
    for i in tqdm(range(0, X.size(0), batch_size)):
        range_end = min(i + batch_size, X.size(0))
        result.append(msdf_at_point(X[i:range_end], centers, scales, Vi))
    return torch.cat(result, dim=0)


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

    centered = (X.view(-1, 1, 3) - centers.view(1, -1, 3)) / scales.view(1, -1, 1)  # N, M, 3
    out_of_cube = torch.any(torch.abs(centered) > 1, dim=-1)
    out_of_cube = torch.all(out_of_cube, dim=-1)
    inside_cube = ~out_of_cube
    result = torch.zeros(X.size(0)).float().to(DEVICE)
    if not torch.any(inside_cube):
        return result
    centered = centered[inside_cube]

    centered_flat = centered.view(-1, 3).float()  # N * M, 3
    weights, indices = find_closest_vertices_indices(centered_flat.cuda())
    indices = indices.view(centered.size(0), centers.size(0), 8, 3)
    weights = weights.view(centered.size(0), centers.size(0), 8)

    values_at_corners = get_values_at_indices(Vi, indices)
    weights = weights.to(values_at_corners.device)
    values = (values_at_corners * weights).sum(dim=-1)
    weights = calculate_weights(X[inside_cube], centers, scales)
    weighted_values = (values * weights).sum(dim=1).float()
    result[inside_cube] = weighted_values
    return result


def calculate_msdf_value(scale: float, points: torch.Tensor, mesh: trimesh.Trimesh) -> torch.Tensor:
    all_points = get_grid_points(points, scale)
    n_points, n_grids, _ = all_points.shape
    f = SDF(mesh.vertices, mesh.faces)
    all_points = all_points.detach().cpu().numpy()
    sdf_values = f(all_points.reshape(-1, 3)).reshape(n_points, n_grids)
    return -torch.tensor(sdf_values, device=DEVICE)


def sample_volume(bounds, resolution, bound_delata=0.1):
    """
    Sample a 3D volume defined by 'bounds' at a given 'resolution'.
    bounds: (min, maxh)
    resolution: int, number of samples per dimension
    bound_delta: float, the delta to add to the bounds to avoid sampling on the boundary.
    Returns a tensor of shape (N, 3) of sampled points.
    """
    delta = (bounds[1] - bounds[0]) * bound_delata
    grid_x = torch.linspace(bounds[0] - delta, bounds[1] + delta, resolution)
    grid_y = torch.linspace(bounds[0] - delta, bounds[1] + delta, resolution)
    grid_z = torch.linspace(bounds[0] - delta, bounds[1] + delta, resolution)
    meshgrid = torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    points = torch.stack(meshgrid, dim=-1).reshape(-1, 3).to(DEVICE)
    return points


def get_grid_points(centers, scales, kernel_size: int = KERNEL_SIZE):
    """
    Get the grid points for the given centers and scales.
    :param centers: Nx3 tensor of centers
    :param scales: N tensor of scales
    :param kernel_size: kernel size used
    :return: NxK*K*Kx3 tensor of grid points
    """
    grid = get_grid(kernel_size=kernel_size)
    grid_flat = grid.reshape(-1, 3).to(centers.device)
    if isinstance(scales, torch.Tensor):
        scales = scales.view(-1, 1, 1)
    all_points = centers.reshape(centers.shape[0], 1, 3) + grid_flat.view(1, -1, 3) * scales
    return all_points


def reconstruct_mesh(Vi, scales, centers, resolution=16, threshold=0.0) -> trimesh.Trimesh:
    grid = sample_volume((-1, 1), resolution)
    Vi = Vi.view(Vi.size(0), KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE)

    grid_sdf = msdf_at_point_batched(grid, centers, scales, Vi)
    grid_sdf = grid_sdf.detach().cpu().numpy().reshape(resolution, resolution, resolution)

    grid_sdf = grid_sdf < threshold
    verts, faces, _, _ = marching_cubes(grid_sdf, level=0.0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh


def reconstruct_from_sdf(sdf, resolution):
    grid = sample_volume((-1, 1), resolution)
    grid_sdf = sdf(grid)
    grid_sdf = grid_sdf.reshape(resolution, resolution, resolution)

    grid_sdf = grid_sdf > 0
    verts, faces, _, _ = marching_cubes(grid_sdf, level=0.0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh
