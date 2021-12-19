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


class ChemblCorpus(pl.LightningDataModule):
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
        chembl_full = pd.read_table(self.data_dir / "chembl_corpus.txt")["Token"]
        chembl_test = chembl_full.sample(frac=0.2, random_state=42)
        chembl_full = chembl_full.drop(
            chembl_test.index
        )  # Make sure the test set is excluded

        if stage == "fit" or stage is None:
            chembl_train, chembl_val = random_split_frac(dataset=chembl_full)
            self.chembl_train = torch.LongTensor(
                self.vocabulary.encode([seq.split(" ") for seq in chembl_train])
            )
            self.chembl_val = torch.LongTensor(
                self.vocabulary.encode([seq.split(" ") for seq in chembl_val])
            )

        if stage == "test" or stage is None:
            self.chembl_test = torch.LongTensor(
                self.vocabulary.encode([seq.split(" ") for seq in chembl_test])
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.chembl_train, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.chembl_val, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.chembl_test, batch_size=self.batch_size)
