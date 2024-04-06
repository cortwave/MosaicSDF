from fire import Fire
import torch
import taichi as ti

from vae.train import VAETrainer, MSDFVAE
from vae.data import MsdfDataModule

from utils.msdf import reconstruct_mesh
from utils.vis import visualize_mesh


def evaluate(data_path: str, weights_path: str):
    ti.init(arch=ti.gpu)
    data_module = MsdfDataModule(data_path)
    trainer: VAETrainer = VAETrainer.load_from_checkpoint(weights_path, repr_shape=data_module.get_representation_shape())
    vae_model: MSDFVAE = trainer.model
    vae_model.eval()

    val_dataloader = data_module.val_dataloader()

    for batch in val_dataloader:
        original_msdf = batch['msdf'].to(trainer.device)
        reconstructed, _ = vae_model(original_msdf)

        original_msdf = val_dataloader.dataset.rescale(original_msdf)
        reconstructed = val_dataloader.dataset.rescale(reconstructed)

        Vi, centers, scales = trainer.separate_sdf_points_scale(original_msdf)
        Vi_hat, centers_hat, scales_hat = trainer.separate_sdf_points_scale(reconstructed)
        for idx in range(len(original_msdf)):
            original_mesh = reconstruct_mesh(Vi[idx], scales[idx], centers[idx])
            reconstructed_mesh = reconstruct_mesh(Vi_hat[idx], scales_hat[idx], centers_hat[idx])
            visualize_mesh(original_mesh)
            visualize_mesh(reconstructed_mesh)


if __name__ == '__main__':
    with torch.no_grad():
        Fire(evaluate)
