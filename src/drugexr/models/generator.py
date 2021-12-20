from typing import Optional
import pytorch_lightning as pl
import torch

import logging
from drugexr.data.chembl_corpus import ChemblCorpus

from drugexr.utils.tensor_ops import print_auto_logged_info


class Generator(pl.LightningModule):
    def __init__(self, vocabulary, embed_size=128, hidden_size=512, lr=1e-3):
        super().__init__()
        self.voc = vocabulary
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = vocabulary.size
        self.lr = lr

        self.embed = torch.nn.Embedding(vocabulary.size, embed_size, device=self.device)
        self.rnn = torch.nn.LSTM(embed_size, hidden_size, num_layers=3, batch_first=True, device=self.device)
        self.linear = torch.nn.Linear(hidden_size, vocabulary.size, device=self.device)
        self.automatic_optimization = False

    def forward(self, x, h):
        output = self.embed(x.unsqueeze(-1))
        output, h_out = self.rnn(output, h)
        output = self.linear(output).squeeze(1)
        return output, h_out

    def init_h(self, batch_size, labels=None):
        h = torch.rand(3, batch_size, 512, device=self.device)
        if labels is not None:
            h[0, batch_size, 0] = labels
        c = torch.rand(3, batch_size, self.hidden_size, device=self.device)
        return (h, c)

    def likelihood(self, target):
        batch_size, seq_len = target.size()
        x = torch.LongTensor([self.voc.tk2ix["GO"]] * batch_size, device=self.device)
        h = self.init_h(batch_size)
        scores = torch.zeros(batch_size, seq_len, device=self.device)
        for step in range(seq_len):
            logits, h = self(x, h)
            logits = logits.log_softmax(dim=-1)
            score = logits.gather(1, target[:, step : step + 1]).squeeze()
            scores[:, step] = score
            x = target[:, step]
        return scores

    def policy_gradient_loss(self, loader: DataLoader) -> None:
        """ """
        opt = self.optimizers()
        for sequence, reward in loader:
            opt.zero_grad()
            score = self.likelihood(sequence)
            loss = score * reward
            loss = -loss.mean()
            self.manual_backward(loss)
            opt.step()

    def sample(self, batch_size):
        # TODO: Maybe extract sample to take a generator model instead of being a method?
        x = torch.LongTensor([self.voc.tk2ix["GO"]] * batch_size, device=self.device)
        h = self.init_h(batch_size)
        sequences = torch.zeros(batch_size, self.voc.max_len, device=self.device).long()
        is_end = torch.zeros(batch_size, device=self.device).bool()

        for step in range(self.voc.max_len):
            logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            x = torch.multinomial(proba, 1).view(-1)
            x[is_end] = self.voc.tk2ix["EOS"]
            sequences[:, step] = x

            end_token = x == self.voc.tk2ix["EOS"]
            is_end = torch.ge(is_end + end_token, 1)
            if (is_end == 1).all():
                break
        return sequences

    def evolve(
        self,
        batch_size: int,
        epsilon: float = 0.01,
        crover: Optional["Generator"] = None,
        mutate: Optional["Generator"] = None,
    ):
        # Start tokens
        x = torch.LongTensor([self.voc.tk2ix["GO"]] * batch_size, device=self.device)
        # Hidden states initialization for exploitation network
        h = self.init_h(batch_size)
        # Hidden states initialization for exploration network
        h2 = self.init_h(batch_size)
        # Initialization of output matrix
        sequences = torch.zeros(batch_size, self.voc.max_len, device=self.device).long()
        # labels to judge and record which sample is ended
        is_end = torch.zeros(batch_size, device=self.device).bool()

        for step in range(self.voc.max_len):
            is_change = torch.rand(1) < 0.5
            if crover is not None and is_change:
                logit, h = crover(x, h)
            else:
                logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            if mutate is not None:
                logit2, h2 = mutate(x, h2)
                ratio = torch.rand(batch_size, 1, device=self.device) * epsilon
                proba = (
                    logit.softmax(dim=-1) * (1 - ratio) + logit2.softmax(dim=-1) * ratio
                )
            # sampling based on output probability distribution
            x = torch.multinomial(proba, 1).view(-1)

            x[is_end] = self.voc.tk2ix["EOS"]
            sequences[:, step] = x

            # Judging whether samples are end or not.
            end_token = x == self.voc.tk2ix["EOS"]
            is_end = torch.ge(is_end + end_token, 1)
            #  If all of the samples generation being end, stop the sampling process
            if (is_end == 1).all():
                break
        return sequences

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.likelihood(batch)
        loss = -loss.mean()
        self.manual_backward(loss)
        self.log("train_loss", loss)
        opt.step()
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.likelihood(batch)
        loss = -loss.mean()
        self.log("val_loss", loss)
        return loss


if __name__ == "__main__":
    from pathlib import Path

    import mlflow
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv
    from torch.utils.data import DataLoader, random_split

    from src.drugexr.config import constants as c
    from src.drugexr.data_structs.vocabulary import Vocabulary

    load_dotenv()

    def train_lstm(
        corpus_path: Path,
        vocabulary_path: Path,
        dev: bool = False,
        n_workers: int = 1,
        epochs: int = 1000,
    ):
        """
        Function to configure training of the LSTM that will be used as a Prior / Agent
        TODO: Update this docstring when it's done.
        """
        if dev:
            from src.drugexr.config import constants as c

            corpus_path = c.PROC_DATA_PATH / "chembl_corpus_DEV_1000.txt"

        logging.info("Loading vocabulary...")
        vocabulary = Vocabulary(vocabulary_path=vocabulary_path)
        logging.info("Creating Generator instance...")
        generator = Generator(vocabulary=vocabulary)

        # logging.info("Creating DataLoader...")
        # chembl = pd.read_table(corpus_path).Token  # Make this more generic?
        # chembl = torch.LongTensor(vocabulary.encode([seq.split(" ") for seq in chembl]))

        # n_samples = len(chembl)
        # train_samples = np.floor(0.9 * n_samples)
        # val_samples = n_samples - train_samples
        # print(train_samples, val_samples)
        # chembl_train, chembl_val = random_split(chembl, [900, 100])

        # train_loader = DataLoader(
        #     chembl_train,
        #     batch_size=64,
        #     shuffle=True,
        #     drop_last=True,
        #     pin_memory=True,
        #     num_workers=n_workers,
        # )
        # val_loader = DataLoader(
        #     chembl_val,
        #     batch_size=64,
        #     shuffle=True,
        #     drop_last=True,
        #     pin_memory=True,
        #     num_workers=n_workers,
        # )

        logging.info("Creating Trainer...")
        trainer = pl.Trainer(gpus=1, log_every_n_steps=50, max_epochs=epochs, fast_dev_run=True)

        logging.info("Initiating ML Flow tracking...")
        mlflow.set_tracking_uri("https://dagshub.com/naisuu/drugex-plus-r.mlflow")
        mlflow.pytorch.autolog()

        datamodule = ChemblCorpus(n_workers=n_workers)
        # datamodule.prepare_data()
        datamodule.setup(stage="fit")

        logging.info("Starting run...")
        with mlflow.start_run() as run:
            trainer.fit(
                model=generator,
                datamodule=datamodule
            )

        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    def sample(nr_samples: int, vocabulary_path: Path, model_ckpt: Path):
        vocabulary = Vocabulary(vocabulary_path=vocabulary_path)
        generator = Generator(vocabulary=vocabulary)
        generator.load_from_checkpoint(model_ckpt, vocabulary=vocabulary)

        encoded_samples = generator.sample(nr_samples)
        decoded_samples = [vocabulary.decode(sample) for sample in encoded_samples]
        print(decoded_samples)

    train_lstm(
        corpus_path=c.PROC_DATA_PATH / "chembl_corpus.txt",
        vocabulary_path=c.PROC_DATA_PATH / "chembl_voc.txt",
        dev=True,
        n_workers=8,
        epochs=100,
    )
