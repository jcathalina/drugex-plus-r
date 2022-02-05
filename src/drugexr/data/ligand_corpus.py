from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from torch.utils.data.dataloader import DataLoader

from drugexr.config.constants import PROC_DATA_PATH
from drugexr.data_structs.vocabulary import Vocabulary
from drugexr.utils.tensor_ops import random_split_frac


class LigandCorpus(pl.LightningDataModule):
    def __init__(
        self,
        vocabulary: Vocabulary,
        data_dir: Path = PROC_DATA_PATH,
        batch_size: int = 512,
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        ligand_full = pd.read_table(self.data_dir / "ligand_corpus.txt")["Token"]

        if stage == "test" or stage is None:
            ligand_test = ligand_full.sample(frac=0.2, random_state=42)
            ligand_full = ligand_full.drop(
                ligand_test.index
            )  # Make sure the test set is excluded
            self.ligand_test = torch.LongTensor(
                self.vocabulary.encode([seq.split(" ") for seq in ligand_test])
            )

        if stage == "fit" or stage is None:
            ligand_train, ligand_val = random_split_frac(dataset=ligand_full)
            self.ligand_train = self.vocabulary.encode(
                [seq.split(" ") for seq in ligand_train]
            )
            self.ligand_val = self.vocabulary.encode(
                [seq.split(" ") for seq in ligand_val]
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.ligand_train, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.ligand_val, batch_size=self.batch_size, pin_memory=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.ligand_test, batch_size=self.batch_size)
