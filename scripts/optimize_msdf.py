import numpy as np
import torch
from fire import Fire
import taichi
import trimesh
from tqdm import tqdm
import point_cloud_utils as pcu

from utils.mesh import load_mesh, farthest_point_sampling
from utils.vis import visualize_mesh
from scipy.spatial import cKDTree
from pysdf import SDF

from config import NUM_GRIDS, DEVICE, KERNEL_SIZE
from utils.msdf import calculate_msdf_value, reconstruct_mesh, msdf_at_point


def calculate_scale(points: torch.Tensor) -> float:
    """
    Calculate the initial scale s which is enough to cover all the mesh. I think that it can be set to the half of the
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
    return max_distance * 0.5


class MSDFOptimizer(torch.nn.Module):
    def __init__(self, Vi, centers, scales, mesh: trimesh.Trimesh, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Vi = torch.nn.Parameter(Vi, requires_grad=True)
        self.centers = torch.nn.Parameter(centers, requires_grad=False)
        self.scales = torch.nn.Parameter(scales, requires_grad=True)
        self.sdf = SDF(mesh.vertices, mesh.faces)
        self.mesh = mesh
        self._sample_mesh()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def _sample_mesh(self, surface_points_count=2048, vicinities=2048):
        points = self.mesh.vertices
        normals = self.mesh.vertex_normals
        indices_surface = np.random.choice(np.arange(points.shape[0]), min(surface_points_count, points.shape[0]), replace=False)
        indices_vicinities = np.random.choice(np.arange(points.shape[0]), min(vicinities, points.shape[0]), replace=False)

        surface = points[indices_surface]
        surface_normals = normals[indices_surface]
        surface_normals = surface_normals / np.linalg.norm(surface_normals, axis=1)[:, None]
        offsets = np.random.normal(0, 0.1, surface_points_count)
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
        loop = tqdm(range(2))
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


def optimize_msdf(input_path: str = 'assets/octopus/model.obj', output_path: str = None):
    """
    Optimize the mesh for MSDF generation.
    :param input_path: path to the input mesh
    :param output_path: path to the output msdf representation
    """
    init_mesh = load_mesh(input_path)

    w_vertices, w_faces = pcu.make_mesh_watertight(init_mesh.vertices, init_mesh.faces)
    mesh = trimesh.Trimesh(vertices=w_vertices, faces=w_faces)

    sampled_points = farthest_point_sampling(mesh.vertices, NUM_GRIDS)
    sampled_points = torch.from_numpy(sampled_points).to(DEVICE)
    init_scale = calculate_scale(sampled_points)
    scales = torch.ones(NUM_GRIDS, device=DEVICE) * init_scale
    initial_msdf_values = calculate_msdf_value(init_scale, sampled_points, mesh)

    msdf_values = initial_msdf_values.view(-1, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE).to(DEVICE)

    msdf_values = msdf_values.view(-1, KERNEL_SIZE ** 3)
    concatenated = torch.cat([msdf_values, sampled_points, scales[:, None]], dim=1)

    torch.save(concatenated, output_path)


if __name__ == '__main__':
    taichi.init(arch=taichi.gpu)
    Fire(optimize_msdf)
