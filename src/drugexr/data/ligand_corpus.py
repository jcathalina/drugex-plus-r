from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data.dataloader import DataLoader

import src.drugexr.config.constants as const
from src.drugexr.data_structs.vocabulary import Vocabulary
from src.drugexr.utils.tensor_ops import random_split_frac


class LigandCorpus(pl.LightningDataModule):
    def __init__(
        self,
        vocabulary: Vocabulary,
        data_dir: Path = const.PROC_DATA_PATH,
        batch_size: int = 32,
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        ligand_full = pd.read_table(self.data_dir / "chembl_corpus.txt")["Token"]
        ligand_test = ligand_full.sample(frac=0.2, random_state=42)
        ligand_full = ligand_full.drop(
            ligand_test.index
        )  # Make sure the test set is excluded

        if stage == "fit" or stage is None:
            ligand_train, ligand_val = random_split_frac(dataset=ligand_full)
            self.ligand_train = torch.LongTensor(
                self.vocabulary.encode([seq.split(" ") for seq in ligand_train])
            )
            self.ligand_val = torch.LongTensor(
                self.vocabulary.encode([seq.split(" ") for seq in ligand_val])
            )

        if stage == "test" or stage is None:
            self.ligand_test = torch.LongTensor(
                self.vocabulary.encode([seq.split(" ") for seq in ligand_test])
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.ligand_train, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.ligand_val, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.ligand_test, batch_size=self.batch_size)
