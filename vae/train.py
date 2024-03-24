from fire import Fire

import pytorch_lightning as pl
import torch

from vae.model import MSDFVAE
from vae.data import MsdfDataModule


class VAETrainer(pl.LightningModule):
    def __init__(self, repr_shape):
        super().__init__()
        self.model = MSDFVAE(representation_shape=repr_shape)

    def _separate_sdf_points_scale(self, x):
        return x[..., :-4], x[..., -4:-1], x[..., -1]

    def forward(self, batch, reduction="mean"):
        x = batch["msdf"]
        x_hat, kl_div = self.model(x)
        sdf, points, scale = self._separate_sdf_points_scale(x)
        x_hat_sdf, x_hat_points, x_hat_scale = self._separate_sdf_points_scale(x_hat)

        sdf_mse = torch.nn.functional.mse_loss(x_hat_sdf, sdf, reduction=reduction)
        points_mse = torch.nn.functional.mse_loss(x_hat_points, points, reduction=reduction)
        scale_mse = torch.nn.functional.mse_loss(x_hat_scale, scale, reduction=reduction)

        kl_div_loss = kl_div
        loss = 0.05 * kl_div_loss + sdf_mse.mean() + 10 * points_mse.mean() + 3 * scale_mse.mean()
        return {
            "loss": loss,
            "sdf_mse": sdf_mse,
            "points_mse": points_mse,
            "scale_mse": scale_mse,
            "kl_div_loss": kl_div_loss,
        }

    def training_step(self, batch, batch_idx):
        losses = self.forward(batch)
        self.log_dict({f'train/{k}': v for k, v in losses.items()}, on_epoch=True, on_step=True, prog_bar=True,
                      logger=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        losses = self.forward(batch)
        self.log_dict({f'val/{k}': v for k, v in losses.items()}, on_epoch=True, on_step=True, prog_bar=True,
                      logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }


def main(data_path: str):
    data_module = MsdfDataModule(data_path)
    training_module = VAETrainer(repr_shape=data_module.get_representation_shape())
    logger = pl.loggers.TensorBoardLogger("logs", name="msdf_vae")
    checkpoints = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="msdf_vae-{epoch:02d}",
        save_top_k=2,
        monitor="val/loss",
        mode="min",
    )
    trainer = pl.Trainer(max_epochs=10, logger=logger, callbacks=[checkpoints], accelerator='gpu')
    trainer.fit(training_module, data_module)


if __name__ == "__main__":
    Fire(main)
