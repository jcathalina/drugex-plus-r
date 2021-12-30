import os
import time
from pathlib import Path
from shutil import copy2

import torch

from drugexr.config.constants import (MODEL_PATH, PROC_DATA_PATH, ROOT_PATH,
                                      TEST_RUN)
from drugexr.data_structs.environment import Environment
from drugexr.data_structs.vocabulary import Vocabulary
from drugexr.models.drugex_r import DrugExR, RewardScheme
from drugexr.models.generator import Generator
from drugexr.models.predictor import Predictor
from drugexr.scoring.ra_scorer import RetrosyntheticAccessibilityScorer
from drugexr.utils import normalization


def main():
    pass


if __name__ == "__main__":
    case = "OBJ4"
    z = "REG"
    alg = "evolve"
    scheme = RewardScheme.PARETO_FRONT

    # construct the environment with three predictors
    keys = ["A1", "A2A", "ERG", "RA_SCORE"]
    A1 = Predictor(path=MODEL_PATH / f"output/single/RF_{z}_CHEMBL226.pkg", type_=z)
    A2A = Predictor(path=MODEL_PATH / f"output/single/RF_{z}_CHEMBL251.pkg", type_=z)
    ERG = Predictor(path=MODEL_PATH / f"output/single/RF_{z}_CHEMBL240.pkg", type_=z)
    RA_SCORER = RetrosyntheticAccessibilityScorer(use_xgb_model=False)
    # TODO: Pass a function that takes a list of Mol objects (or use the raw smiles if this turns out to be faster)

    # Chose the desirability function
    objs = [A1, A2A, ERG, RA_SCORER]
    n_objs = len(objs)

    if scheme == RewardScheme.WEIGHTED_SUM:
        mod1 = normalization.ClippedScore(lower_x=3, upper_x=10)
        mod2 = normalization.ClippedScore(lower_x=10, upper_x=3)
        ths = [0.5] * n_objs
    elif scheme == RewardScheme.PARETO_FRONT:
        mod1 = normalization.ClippedScore(lower_x=3, upper_x=6.5)
        mod2 = normalization.ClippedScore(lower_x=10, upper_x=6.5)
        ths = [0.99] * n_objs
    else:
        raise ValueError(
            f"Valid reward schemes include {RewardScheme.PARETO_FRONT}, {RewardScheme.WEIGHTED_SUM}"
        )

    mods = [mod1, mod1, mod2]
    env = Environment(objs=objs, mods=mods, keys=keys, ths=ths)

    root_suffix = f"output/{alg}_{case}_{scheme}_{time.strftime('%y%m%d_%H%M%S', time.localtime())}/"
    root: Path = MODEL_PATH / root_suffix
    os.mkdir(root)

    copy2(ROOT_PATH / "src/drugexr/models/drugex_r.py", root)
    copy2(ROOT_PATH / "src/drugexr/training/train.py", root)

    pr_path: Path = MODEL_PATH / "output/rnn/lstm_chembl_R"
    ft_path: Path = MODEL_PATH / "output/rnn/lstm_ligand_R"

    voc = Vocabulary(vocabulary_path=PROC_DATA_PATH / "chembl_voc.txt")
    agent = Generator(voc)
    agent.load_state_dict(torch.load(ft_path.with_suffix(".pkg")))

    prior = Generator(voc)
    prior.load_state_dict(torch.load(pr_path.with_suffix(".pkg")))

    crover = Generator(voc)
    crover.load_state_dict(torch.load(ft_path.with_suffix(".pkg")))

    learner = DrugExR(
        agent=agent, prior=prior, xover_net=crover, mutator_net=prior, environment=env
    )

    outfile = "%s_%s_%s_%s" % (alg, learner.scheme, z, case)
    outfile += "_%.0e" % learner.epsilon

    output_path = root / outfile

    epochs = 50 if TEST_RUN else 10_000
    learner.fit(output_path=output_path, epochs=epochs, interval=5)
