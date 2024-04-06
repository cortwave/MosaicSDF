import logging

import numpy as np
import torch
from fire import Fire
import trimesh
from tqdm import tqdm
import point_cloud_utils as pcu

from utils.mesh import load_mesh, farthest_point_sampling
from utils.vis import visualize_mesh
from scipy.spatial import cKDTree
from pysdf import SDF

from config import NUM_GRIDS, DEVICE, KERNEL_SIZE
from utils.msdf import calculate_msdf_value, reconstruct_mesh, msdf_at_point, reconstruct_from_sdf
from utils import msdf as msdf_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_scale(points: torch.Tensor) -> float:
    """
    Calculate the initial scale s which is enough to cover all the mesh. I think that it can be set to the
    maximum distance between two neighboring points.
    :param points: the points of the mesh
    :return: the scale of the mesh
    """
    points = points.cpu().numpy()
    kdtree = cKDTree(points)
    max_distance = 0
    distances, _ = kdtree.query(points, k=2)
    for distance in distances:
        max_distance = max(max_distance, distance[1])
    return max_distance


class MSDFOptimizer(torch.nn.Module):
    def __init__(self,
                 Vi: torch.Tensor,
                 centers: torch.Tensor,
                 scales: torch.Tensor,
                 mesh: trimesh.Trimesh,
                 n_steps: int = 1000):
        super().__init__()
        self.Vi = torch.nn.Parameter(Vi, requires_grad=True)
        self.centers = torch.nn.Parameter(centers, requires_grad=True)
        self.scales = torch.nn.Parameter(scales, requires_grad=True)
        self.sdf = SDF(mesh.vertices, mesh.faces)
        self.mesh = mesh
        self.n_steps = n_steps
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

    def _sample_mesh(self, surface_points_count=2048, vicinities_count=2048):
        points = self.mesh.vertices
        normals = self.mesh.vertex_normals
        indices_surface = np.random.choice(np.arange(points.shape[0]), min(surface_points_count, points.shape[0]),
                                           replace=False)
        indices_vicinities = np.random.choice(np.arange(points.shape[0]), min(vicinities_count, points.shape[0]),
                                              replace=False)

        surface = points[indices_surface]
        surface_normals = normals[indices_surface]
        surface_normals = surface_normals / np.linalg.norm(surface_normals, axis=1)[:, None]
        offsets = np.random.normal(0, 0.1, vicinities_count)
        vicinities = points[indices_vicinities] + normals[indices_vicinities] * offsets[:, None]
        vicinities_sdf = self.sdf(vicinities)
        surface, surface_normals, vicinities, vicinities_sdf = map(
            lambda x: torch.from_numpy(x).to(DEVICE).float(), (surface, surface_normals, vicinities, vicinities_sdf))
        return surface, surface_normals, vicinities, vicinities_sdf

    def optimizer_step(self):
        self.optimizer.zero_grad()

        surface, surface_normals, vicinities, vicinities_sdf = self._sample_mesh()

        surface = torch.nn.Parameter(surface.clone(), requires_grad=True)
        surface_pred = msdf_at_point(surface, self.centers, self.scales, self.Vi)
        vicinities_pred = msdf_at_point(vicinities, self.centers, self.scales, self.Vi)

        surface_loss = surface_pred.abs().mean()
        vicinitties_loss = (vicinities_pred - vicinities_sdf).abs().mean()

        gradient = torch.autograd.grad(outputs=surface_pred, inputs=surface, grad_outputs=torch.ones_like(surface_pred),
                                       create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_loss = (surface_normals - gradient).abs().mean()

        loss = surface_loss + vicinitties_loss + 0.1 * gradient_loss

        self.optimizer.step()
        return loss.item()

    def optimize(self):
        loop = tqdm(range(self.n_steps))
        for _ in loop:
            loss = self.optimizer_step()
            loop.set_description(f'loss: {loss:.5f}')


def chamfer_distance(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh) -> float:
    """
    Calculates the symmetric Chamfer distance between two Trimesh meshes.

    Args:
        mesh1: The first mesh.
        mesh2: The second mesh.

    Returns:
        The Chamfer distance between the two meshes.
    """

    sample_points_1 = farthest_point_sampling(mesh1.vertices, 1000)  # Sample from mesh 1
    sample_points_2 = farthest_point_sampling(mesh2.vertices, 1000)  # Sample from mesh 2
    visualize_mesh(sample_points_1)
    visualize_mesh(sample_points_2)
    average_distance = pcu.hausdorff_distance(sample_points_1, sample_points_2)
    return average_distance


def optimize_msdf(input_path: str = 'assets/octopus/model.obj',
                  output_path: str = None,
                  visualize: bool = False,
                  resolution: int = 32,
                  with_optimization: bool = False):
    """
    Optimize the mesh for MSDF generation.
    :param input_path: path to the input mesh
    :param output_path: path to the output msdf representation
    :param visualize: whether to visualize the mesh
    :param resolution: resolution of the grid for marching cubes
    :param with_optimization: whether to optimize the msdf values after initialization
    """
    init_mesh = load_mesh(input_path)

    if visualize:
        visualize_mesh(init_mesh, name="init_mesh")

    w_vertices, w_faces = pcu.make_mesh_watertight(init_mesh.vertices, init_mesh.faces)
    mesh = trimesh.Trimesh(vertices=w_vertices, faces=w_faces)
    if visualize:
        visualize_mesh(mesh, name="watertight_mesh")

    sdf_gt = SDF(mesh.vertices, mesh.faces)

    sampled_points = farthest_point_sampling(mesh.vertices, NUM_GRIDS)
    sampled_points = torch.from_numpy(sampled_points).to(DEVICE)
    init_scale = calculate_scale(sampled_points)
    logger.info(f'Initial scale: {init_scale}')

    scales = torch.ones(NUM_GRIDS, device=DEVICE).float() * init_scale
    grid_points = msdf_utils.get_grid_points(sampled_points, init_scale).detach().cpu().numpy().reshape(-1, 3)
    initial_msdf_values = calculate_msdf_value(init_scale, sampled_points, mesh)

    msdf_values = initial_msdf_values.view(-1, KERNEL_SIZE ** 3).to(DEVICE)

    if with_optimization:
        optimizer = MSDFOptimizer(msdf_values.view(-1, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE), sampled_points, scales,
                                  mesh)
        optimizer.optimize()
        optimized_msdf_values = optimizer.Vi.detach().cpu()
        optimized_scales = optimizer.scales.detach().cpu()
        optimized_centers = optimizer.centers.detach().cpu()
        n_grids = optimized_msdf_values.size(0)
        if visualize:
            optimized_mesh = reconstruct_mesh(optimized_msdf_values,
                                              optimized_scales,
                                              optimized_centers,
                                              resolution=resolution)
            visualize_mesh(optimized_mesh, name="optimized_mesh")

        concatenated = torch.cat([optimized_msdf_values.view(n_grids, -1),
                                  optimized_centers.view(n_grids, 3),
                                  optimized_scales[:, None]], dim=1)
    else:
        concatenated = torch.cat([msdf_values, sampled_points, scales[:, None]], dim=1)
    torch.save(concatenated, output_path)

    if visualize:
        reconstructed_mesh = reconstruct_mesh(msdf_values, scales, sampled_points, resolution=resolution)
        reconstructed_gt = reconstruct_from_sdf(sdf_gt, resolution=resolution)
        visualize_mesh(grid_points, name="grid_points")
        visualize_mesh(reconstructed_gt, name="reconstructed_gt")
        visualize_mesh(reconstructed_mesh, name="reconstructed_mesh")


if __name__ == '__main__':
    Fire(optimize_msdf)
