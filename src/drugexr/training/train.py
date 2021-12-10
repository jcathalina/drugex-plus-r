import getopt
import os
import sys
import time
from pathlib import Path
from shutil import copy2

import torch

from src.drugexr.config.constants import MODEL_PATH, PROC_DATA_PATH, ROOT_PATH, TEST_RUN
from src.drugexr.data_structs.environment import Environment
from src.drugexr.data_structs.vocabulary import Vocabulary
from src.drugexr.models.drugex_v2 import DrugExV2
from src.drugexr.models.generator import Generator
from src.drugexr.models.predictor import Predictor
from src.drugexr.utils import normalization

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "a:e:b:g:c:s:z:")
    OPT = dict(opts)
    os.environ["CUDA_VISIBLE_DEVICES"] = OPT["-g"] if "-g" in OPT else "0"
    case = OPT["-c"] if "-c" in OPT else "OBJ3"
    z = OPT["-z"] if "-z" in OPT else "REG"
    alg = OPT["-a"] if "-a" in OPT else "evolve"
    scheme = OPT["-s"] if "-s" in OPT else "PR"

    # construct the environment with three predictors
    # keys = ['A1', 'A2A', 'ERG']
    keys = ["A1"]
    A1 = Predictor(MODEL_PATH / f"output/single/RF_{z}_CHEMBL226.pkg", type_=z)
    # A2A = Predictor('output/env/RF_%s_CHEMBL251.pkg' % z, type=z)
    # ERG = Predictor('output/env/RF_%s_CHEMBL240.pkg' % z, type=z)

    # Chose the desirability function
    # objs = [A1, A2A, ERG]
    objs = [A1]

    if scheme == "WS":
        mod1 = normalization.ClippedScore(lower_x=3, upper_x=10)
        mod2 = normalization.ClippedScore(lower_x=10, upper_x=3)
        ths = [0.5] * len(objs)  # 3
    else:
        mod1 = normalization.ClippedScore(lower_x=3, upper_x=6.5)
        mod2 = normalization.ClippedScore(lower_x=10, upper_x=6.5)
        ths = [0.99] * len(objs)  # 3

    mods = [mod1, mod1, mod2] if case == "OBJ3" else [mod2, mod1, mod2]
    env = Environment(objs=objs, mods=mods, keys=keys, ths=ths)

    root_suffix = "output/%s_%s_%s_%s/" % (
        alg,
        case,
        scheme,
        time.strftime("%y%m%d_%H%M%S", time.localtime()),
    )
    root: Path = MODEL_PATH / root_suffix
    os.mkdir(root)
    copy2(ROOT_PATH / "src/drugexr/models/drugex_v2.py", root)
    copy2(ROOT_PATH / "src/drugexr/training/train.py", root)

    pr_path: Path = MODEL_PATH / "output/rnn/lstm_chembl"
    ft_path: Path = MODEL_PATH / "output/rnn/lstm_ligand_T"

    voc = Vocabulary(vocabulary_path=PROC_DATA_PATH / "chembl_voc.txt")
    agent = Generator(voc)
    agent.load_state_dict(torch.load(ft_path.with_suffix(".pkg")))

    prior = Generator(voc)
    prior.load_state_dict(torch.load(pr_path.with_suffix(".pkg")))

    crover = Generator(voc)
    crover.load_state_dict(torch.load(ft_path.with_suffix(".pkg")))
    learner = DrugExV2(agent, env, prior, crover)

    learner.epsilon = learner.epsilon if "-e" not in OPT else float(OPT["-e"])
    learner.penalty = learner.penalty if "-b" not in OPT else float(OPT["-b"])
    learner.scheme = learner.scheme if scheme is None else scheme

    outfile = "%s_%s_%s_%s" % (alg, learner.scheme, z, case)
    outfile += "_%.0e" % learner.epsilon

    learner.out = root / outfile

    epochs = 50 if TEST_RUN else 10_000
    learner.fit(epochs=epochs)
