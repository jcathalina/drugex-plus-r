import getopt
import logging
import os
import pathlib
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.drugexr.config.constants import MODEL_PATH, PROC_DATA_PATH
from src.drugexr.data_structs.vocabulary import Vocabulary
from src.drugexr.models.generator import Generator


def guarantee_path(dir_: pathlib.Path) -> pathlib.Path:
    """
    TODO: Extract to utils
    Helper function that guarantees that a path exists by creating all necessary (nested) directories
    if necessary and returns that path.
    :param dir_: Path to the directory that needs to be guaranteed
    :return: Path that was passed to the function, after guaranteeing its existence.
    """
    if not pathlib.Path.exists(dir_):
        pathlib.Path(dir_).mkdir(parents=True)
    return dir_


def pretrain_rnn(is_lstm: bool = True):
    voc = Vocabulary(vocabulary_path=pathlib.Path(PROC_DATA_PATH / "chembl_voc.txt"))

    out_dir = MODEL_PATH / "output/rnn"
    if not pathlib.Path.exists(out_dir):
        logging.info(f"Creating directories to store pretraining output @ '{out_dir}'")
        pathlib.Path(out_dir).mkdir(parents=True)

    if is_lstm:
        netP_path = out_dir / "lstm_chembl_T"
        netE_path = out_dir / "lstm_ligand_T"
    else:
        netP_path = out_dir / "gru_chembl"
        netE_path = out_dir / "gru_ligand"

    prior = Generator(voc, is_lstm=is_lstm)
    if not os.path.exists(netP_path.with_suffix(".pkg")):
        chembl = pd.read_table(PROC_DATA_PATH / "chembl_corpus.txt").Token
        chembl = torch.LongTensor(voc.encode([seq.split(" ") for seq in chembl]))
        chembl = DataLoader(chembl, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        prior.fit(chembl, out=netP_path, epochs=50)
    prior.load_state_dict(torch.load(netP_path.with_suffix(".pkg")))

    # explore = model.Generator(voc)
    df = pd.read_table(PROC_DATA_PATH / "ligand_corpus.txt").drop_duplicates("Smiles")
    valid = df.sample(len(df) // 10).Token
    train = df.drop(valid.index).Token
    # explore.load_state_dict(torch.load(netP_path + '.pkg'))

    train = torch.LongTensor(voc.encode([seq.split(" ") for seq in train]))
    train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    valid = torch.LongTensor(voc.encode([seq.split(" ") for seq in valid]))
    valid = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=True)
    print("Fine tuning progress begins to be trained...")

    prior.fit(train, loader_valid=valid, out=netE_path, epochs=10, lr=lr)
    print("Fine tuning progress training is finished...")


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "g:m:")
    OPT = dict(opts)
    lr = 1e-4
    torch.set_num_threads(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if "-g" not in OPT else OPT["-g"]
    BATCH_SIZE = 512
    is_lstm = opts["-m"] if "-m" in OPT else True
    pretrain_rnn(is_lstm=is_lstm)
