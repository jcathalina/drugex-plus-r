import pandas as pd
import torch
import os
from tqdm import tqdm
from rdkit import rdBase

from src.drugexr.config.constants import PROC_DATA_PATH, MODEL_PATH, TEST_RUN
from src.drugexr.data_structs.environment import Environment
from src.drugexr.data_structs.vocabulary import Vocabulary
from src.drugexr.models.generator import Generator
from src.drugexr.models.predictor import Predictor
from src.drugexr.utils.normalization import ClippedScore

rdBase.DisableLog("rdApp.error")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def sampling(netG_path, size=10_000):
    """
    sampling a series of tokens squentially for molecule generation
    Args:
        netG_path (str): The file path of generator.
        size (int): The number of molecules required to be generated.

    Returns:
        smiles (List): A list of generated SMILES-based molecules
    """
    batch_size = 250
    samples = []
    voc = Vocabulary(vocabulary_path=PROC_DATA_PATH / "chembl_voc.txt")
    netG = Generator(voc)
    netG.load_state_dict(torch.load(netG_path))
    batch = size // batch_size
    mod = size % batch_size
    for i in tqdm(range(batch + 1)):
        if i == 0:
            if mod == 0:
                continue
            tokens = netG.sample(batch)
        else:
            tokens = netG.sample(batch_size)
        smiles = [voc.decode(s) for s in tokens]
        samples.extend(smiles)
    return samples


if __name__ == "__main__":
    SAMPLE_SIZE = 1000 if TEST_RUN else 10_000

    for z in ["REG"]:
        # Construct the environment with three predictors and desirability functions
        # keys = ["A1", "A2A", "ERG"]
        keys = ["A1"]
        env_model_path = MODEL_PATH / "output/single"
        A1 = Predictor(env_model_path / f"RF_{z}_CHEMBL226.pkg", type_=z)
        # A2A = Predictor('output/env/RF_%s_CHEMBL251.pkg' % z, type_=z)
        # ERG = Predictor('output/env/RF_%s_CHEMBL240.pkg' % z, type_=z)
        mod1 = ClippedScore(lower_x=4, upper_x=6.5)
        mod2 = ClippedScore(lower_x=9, upper_x=6.5)
        mod3 = ClippedScore(lower_x=7.5, upper_x=5)
        # objs = [A1, A2A, ERG]
        objs = [A1]

        lstm_ligand_model_path = MODEL_PATH / "output/rnn/lstm_ligand_T.pkg"
        lstm_chembl_model_path = MODEL_PATH / "output/rnn/lstm_chembl.pkg"

        benchmark_output_path = MODEL_PATH / "output/benchmark"
        if not benchmark_output_path.exists():
            benchmark_output_path.mkdir()



        for case in ["OBJ1", "OBJ3"]:
            if case == "OBJ3":
                mods = [mod1, mod1, mod3]
            else:
                mods = [mod2, mod1, mod3]

            models = {
                lstm_ligand_model_path: benchmark_output_path / f"FINE-TUNE_{z}_{case}.tsv",
                lstm_chembl_model_path: benchmark_output_path / f"PRE-TRAIN_{z}_{case}.tsv",
            }

            env = Environment(objs=objs, mods=mods, keys=keys)

            # for input, output in models.items():
            #     df = pd.DataFrame()
            #     df["Smiles"] = sampling(netG_path=input, size=SAMPLE_SIZE)
            #     scores = env(df["Smiles"], is_smiles=True)
            #     df.to_csv(output, index=False, sep="\t")

            df = pd.DataFrame()
            df["Smiles"] = sampling(netG_path=lstm_chembl_model_path, size=SAMPLE_SIZE)
            scores = env(df["Smiles"], is_smiles=True)
            df.to_csv(models[lstm_chembl_model_path], index=False, sep="\t")

