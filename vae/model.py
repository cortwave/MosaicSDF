from typing import Tuple

from torch import nn
import torch


class MSDFVAE(nn.Module):
    def __init__(self,
                 representation_shape: Tuple[int, int],
                 hidden_size: int = 512,
                 dim_feedforward: int = 512,
                 nhead=4,
                 latent_size=16,
                 num_layers=2):
        super().__init__()
        self.first_layer = nn.Linear(representation_shape[1], hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward, activation=torch.nn.functional.gelu)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_size)
        )
        self.encoder_sigma = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_size)
        )
        self.predecoder = nn.Linear(latent_size, hidden_size)
        decoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward, activation=torch.nn.functional.gelu)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.final_layer = nn.Linear(hidden_size, representation_shape[1])

    @staticmethod
    def reparametrization(mu, sigma):
        std = torch.exp(0.5 * sigma)
        target_dist = torch.distributions.Normal(0, 1)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        kl_div = torch.distributions.kl.kl_divergence(dist, target_dist).mean()
        return z, kl_div

    def encode(self, x):
        # print('x nan', torch.any(torch.isnan(x)))
        x = self.first_layer(x)
        # print('x2 nan', torch.any(torch.isnan(x)))
        z = self.encoder(x)
        # print('z nan', torch.any(torch.isnan(z)))
        mu = self.encoder_mu(z)
        # print('mu nan', torch.any(torch.isnan(mu)))
        sigma = self.encoder_sigma(z)
        # print('sigma nan', torch.any(torch.isnan(sigma)))
        return mu, sigma

    def decode(self, z):
        z = self.predecoder(z)
        x_hat = self.decoder(z)
        x_hat = self.final_layer(x_hat)
        return x_hat

    def forward(self, x: torch.Tensor):
        mu, sigma = self.encode(x)
        z, kl_div = self.reparametrization(mu, sigma)
        x_hat = self.decode(z)
        return x_hat, kl_div
