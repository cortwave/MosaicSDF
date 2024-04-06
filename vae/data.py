from typing import List
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class MsdfDataset(Dataset):
    def __init__(self, samples: List[Path]):
        self.samples = samples
        self.mean = torch.load('mean.pt').view(1, -1)
        self.std = torch.load('std.pt').view(1, -1)
        # replace 0 std with 1
        self.std[self.std == 0] = 1
        self.std[self.std != 1] = 1

    def __len__(self):
        return len(self.samples)

    def rescale(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        msdf_representation = torch.load(sample)

        permutation = torch.randperm(msdf_representation.size(0))
        msdf_representation = msdf_representation[permutation]
        msdf_representation = (msdf_representation - self.mean) / self.std

        return {
            "path": str(sample),
            "msdf": msdf_representation.float()
        }


class MsdfDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size=64, train_ratio=0.9):
        super().__init__()
        self.batch_size = batch_size
        samples = list(sorted(Path(data_path).rglob("*.pt")))
        self.train_samples = samples[:int(len(samples) * train_ratio)]
        self.val_samples = samples[int(len(samples) * train_ratio):]

    def get_representation_shape(self):
        sample = torch.load(self.train_samples[0])
        return sample.size()

    def setup(self, stage: str = None):
        pass

    def train_dataloader(self):
        dataset = MsdfDataset(self.train_samples)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self):
        dataset = MsdfDataset(self.val_samples)
        return DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=8,
        )

    def test_dataloader(self):
        pass


if __name__ == '__main__':
    from tqdm import tqdm
    from os import remove
    data_path = Path('msdf_embeddings')
    files = list(data_path.rglob('*.pt'))

    mean = torch.zeros(7 * 7 * 7 + 3 + 1, dtype=torch.float64)
    std = torch.zeros(7 * 7 * 7 + 3 + 1, dtype=torch.float64)
    count = 0

    for file in tqdm(files):
        data = torch.load(file)
        file_mean = data.mean(dim=0)
        file_std = data.std(dim=0)

        print(file_mean.mean(), file_std.mean(), mean.mean())
        if file_mean.mean().abs() > 0.1:
            print(file)
            remove(file)
            continue

        std += file_std
        mean += file_mean
        count += 1

    mean /= count
    std /= count

    torch.save(mean, 'mean.pt')
    torch.save(std, 'std.pt')
