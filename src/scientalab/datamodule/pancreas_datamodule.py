from pathlib import Path

import lightning as L
import scanpy as sc
import torch
from torch.utils import data
import numpy as np

from scientalab.datamodule import pancreas_dataset


class PancreasDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        invariants: list[str] = ["celltype"],
        spurious: list[str] = ["batch"],
        batch_size: int = 256,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_preprocessed = self.data_dir / "preprocessed.h5ad"
        self.invariants = invariants
        self.spurious = spurious
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        data = sc.read(
            self.data_dir / 'raw_data.h5ad',
            backup_url="https://www.dropbox.com/s/qj1jlm9w10wmt0u/pancreas.h5ad?dl=1",
        )
        sc.pp.filter_genes(data, min_cells=10)
        sc.pp.filter_cells(data, min_genes=200)
        sc.pp.normalize_total(data)
        sc.pp.log1p(data)
        np.nan_to_num(data.X, copy=False)

        data.write_h5ad(self.data_preprocessed)

    def setup(self, stage: str) -> None:
        full_data = pancreas_dataset.PancreasDataset(
            data_path=self.data_preprocessed, invariants=self.invariants, spurious=self.spurious
        )
        self.data_dim, self.invariant_dim, self.spurious_dim = (
            full_data.data_dim,
            full_data.invariant_dim,
            full_data.spurious_dim,
        )
        self.train, self.val, self.test = data.random_split(
            full_data, lengths=[0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.val, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.test, batch_size=self.batch_size, shuffle=False)
    
    def predict_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.test, batch_size=1, shuffle=False)
