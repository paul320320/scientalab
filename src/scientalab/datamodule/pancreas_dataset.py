import scanpy as sc
import torch
from torch.utils import data

from scientalab import utils


class PancreasDataset(data.Dataset):
    def __init__(
        self, data_path, invariants: list[str] = ["celltype"], spurious: list[str] = ["batch"], label: str = "celltype"
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.data = sc.read(self.data_path)
        self.label = label
        self.invariants = invariants
        self.spurious = spurious
        self.invariants_vocabularies = {
            invariant: self.data.obs[invariant].unique().tolist() for invariant in self.invariants
        }
        self.spurious_vocabularies = {spurious: self.data.obs[spurious].unique().tolist() for spurious in self.spurious}
        self.spurious_dim = sum(len(self.spurious_vocabularies[spurious]) for spurious in self.spurious)
        self.invariant_dim = sum(len(self.invariants_vocabularies[invariant]) for invariant in self.invariants)
        self.data_dim = self.invariant_dim + self.spurious_dim + self.data.X.shape[1]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> dict[str, torch.Tensor | list[str]]:
        x = torch.Tensor(self.data.X[index])
        invariant_data = torch.Tensor(
            *[
                utils.one_hot_encoding(
                    self.data.obs[invariant].iloc[index], vocabulary=self.invariants_vocabularies[invariant]
                )
                for invariant in self.invariants
            ]
        )
        spurious_data = torch.Tensor(
            *[
                utils.one_hot_encoding(
                    self.data.obs[spurious].iloc[index], vocabulary=self.spurious_vocabularies[spurious]
                )
                for spurious in self.spurious
            ]
        )
        invariants_label = {invariant: self.data.obs[invariant].iloc[index] for invariant in self.invariants}
        spurious_label = {spurious: self.data.obs[spurious].iloc[index] for spurious in self.spurious}

        features = {"x": torch.cat([x, invariant_data, spurious_data])}
        features.update(invariants_label)
        features.update(spurious_label)

        return features