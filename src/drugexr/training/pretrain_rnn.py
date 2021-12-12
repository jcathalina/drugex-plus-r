import getopt
import logging
import os
import pathlib
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.drugexr.config.constants import MODEL_PATH, PROC_DATA_PATH, TEST_RUN
from src.drugexr.data.preprocess import logger
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
#
#
# def pretrain_rnn(is_lstm: bool = True, epochs: int = 50, epochs_ft: int = 10, lr: float = 1e-4):
#     voc = Vocabulary(vocabulary_path=pathlib.Path(PROC_DATA_PATH / "chembl_voc.txt"))
#
#     out_dir = MODEL_PATH / "output/rnn"
#     if not pathlib.Path.exists(out_dir):
#         logging.info(f"Creating directories to store pretraining output @ '{out_dir}'")
#         pathlib.Path(out_dir).mkdir(parents=True)
#
#     if is_lstm:
#         netP_path = out_dir / "lstm_chembl_R"
#         netE_path = out_dir / "lstm_ligand_R"
#     else:
#         netP_path = out_dir / "gru_chembl_R"
#         netE_path = out_dir / "gru_ligand_R"
#
#     prior = Generator(voc, is_lstm=is_lstm)
#     if not os.path.exists(netP_path.with_suffix(".pkg")):
#         logger.info("No pre-existing model found, starting training process...")
#         chembl = pd.read_table(PROC_DATA_PATH / "chembl_corpus.txt").Token
#         chembl = torch.LongTensor(voc.encode([seq.split(" ") for seq in chembl]))
#         chembl = DataLoader(chembl, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#         prior.fit(chembl, out=netP_path, epochs=epochs)
#     prior.load_state_dict(torch.load(netP_path.with_suffix(".pkg")))
#
#     # explore = model.Generator(voc)
#     df = pd.read_table(PROC_DATA_PATH / "ligand_corpus.txt").drop_duplicates("Smiles")
#     valid = df.sample(len(df) // 10).Token
#     train = df.drop(valid.index).Token
#     # explore.load_state_dict(torch.load(netP_path + '.pkg'))
#
#     train = torch.LongTensor(voc.encode([seq.split(" ") for seq in train]))
#     train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
#
#     valid = torch.LongTensor(voc.encode([seq.split(" ") for seq in valid]))
#     valid = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=True)
#     print("Fine tuning progress begins to be trained...")
#
#     prior.fit(train, loader_valid=valid, out=netE_path, epochs=epochs_ft, lr=lr)
#     print("Fine tuning progress training is finished...")


if __name__ == "__main__":
    import mlflow
    from dotenv import load_dotenv

    load_dotenv()

    mlflow.start_run()
    mlflow.set_tracking_uri("https://dagshub.com/naisuu/drugex-plus-r.mlflow")

    voc = Vocabulary(vocabulary_path=pathlib.Path(PROC_DATA_PATH / "chembl_voc.txt"))
    out_dir = MODEL_PATH / "output/rnn"
    netP_path = out_dir / "lstm_chembl_R"
    netE_path = out_dir / "lstm_ligand_R"
    prior = Generator(vocabulary=voc)
    chembl = pd.read_table(PROC_DATA_PATH / "chembl_corpus_DEV_1000.txt").Token
    chembl = torch.LongTensor(voc.encode([seq.split(" ") for seq in chembl]))
    chembl = DataLoader(chembl, batch_size=512, shuffle=True, drop_last=True)

    mlflow.pytorch.autolog()

    prior.fit(chembl, out=netP_path, epochs=5)

    # opts, args = getopt.getopt(sys.argv[1:], "g:m:")
    # OPT = dict(opts)
    # torch.set_num_threads(1)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" if "-g" not in OPT else OPT["-g"]
    # BATCH_SIZE = 512
    # is_lstm = opts["-m"] if "-m" in OPT else True
    #
    # epochs = 1 if TEST_RUN else 50
    # epochs_ft = 1 if TEST_RUN else 10
    # pretrain_rnn(is_lstm=is_lstm, epochs=epochs, epochs_ft=epochs_ft)
