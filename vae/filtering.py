from fire import Fire
import taichi as ti
import torch
from tqdm import tqdm
from pathlib import Path
from utils.msdf import reconstruct_mesh
from utils.vis import visualize_mesh


from vae.train import VAETrainer
from vae.data import MsdfDataModule
from config import KERNEL_SIZE


def main(weights_path: str, data_path: str):
    data_module = MsdfDataModule(data_path)
    vae_trainer = VAETrainer.load_from_checkpoint(weights_path, repr_shape=data_module.get_representation_shape())
    vae_trainer.eval()

    sample2loss = {}
    for dataloader in [data_module.train_dataloader(), data_module.val_dataloader()]:
        for batch in tqdm(dataloader):
            batch['msdf'] = batch['msdf'].to(vae_trainer.device)
            with torch.no_grad():
                losses = vae_trainer.forward(batch, reduction="none")
            mse_loss = losses['mse_loss'].mean(dim=(1, 2))
            for idx, (name, l) in enumerate(zip(batch['path'], mse_loss)):
                sample2loss[name] = l.item()
                if l > 0.5:
                    msdf = batch['msdf'][idx]
                    print(name, l.item(), msdf.min(), msdf.max(), msdf.mean())

    print(f"Mean loss: {sum(sample2loss.values()) / len(sample2loss)}")
    torch.save(sample2loss, 'sample2loss.pt')


def visualize_outliers(path='sample2loss.pt'):
    ti.init(arch=ti.gpu)
    sample2loss = torch.load(path)
    sorted_samples = sorted(sample2loss.items(), key=lambda x: x[1], reverse=True)
    for sample, loss in sorted_samples:
        print(loss)
        tensor = torch.load(sample)
        print(tensor)
        print(tensor.max(), tensor.min(), tensor.mean())
        Vi = tensor[:, :KERNEL_SIZE**3].contiguous().float()
        points = tensor[:, KERNEL_SIZE**3:KERNEL_SIZE**3+3].contiguous().float()
        scale = tensor[:, -1].contiguous().float()
        mesh = reconstruct_mesh(Vi, scale, points)
        visualize_mesh(mesh)


if __name__ == '__main__':
    Fire(main)
