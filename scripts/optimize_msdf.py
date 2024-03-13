from fire import Fire
import trimesh

from utils.mesh import load_mesh, sample_points
from utils.vis import visualize_mesh
from config import NUM_GRIDS


def optimize_msdf(input_path: str = 'assets/octopus/model.obj', output_path: str = None):
    """
    Optimize the mesh for MSDF generation.
    :param input_path: path to the input mesh
    :param output_path: path to the output msdf representation
    """
    mesh = load_mesh(input_path)

    sampled_points = sample_points(mesh, NUM_GRIDS)
    visualize_mesh(sampled_points)


if __name__ == '__main__':
    Fire(optimize_msdf)
