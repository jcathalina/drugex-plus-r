""" module containing the Policy Gradient model that all other reinforcement learning framework models inherit from """
from pathlib import Path

import torch

import src.drugexr.utils.tensor_ops
from src.drugexr.data_structs.environment import Environment
from src.drugexr.models.generator import Generator


class PGLearner(object):
    """
    Reinforcement learning framework with policy gradient.
    This class is the base structure for all policy gradient-based  deep reinforcement learning models.

    Arguments:
        agent (Generator): The agent which generates the desired molecules
        env (Environment): The environment which provides the reward and determines
                         if the generated molecule is valid and desired.
        prior: The auxiliary model which is defined differently in each methods.
        memory: TODO
        mean_func: TODO
    """

    def __init__(
        self,
        agent: Generator,
        env: Environment,
        prior=None,
        memory=None,
        mean_func="geometric",
    ):
        self.replay = 10
        self.agent = agent
        self.prior = prior
        self.batch_size = 64  # * 4
        self.n_samples = 128  # * 8
        self.env = env
        self.epsilon = 1e-3
        self.penalty = 0
        self.scheme = "PR"
        self.out: Path = None  # TODO: What's the point of this one
        self.memory = memory
        # mean_func: which function to use for averaging: 'arithmetic' or 'geometric'
        self.mean_func = mean_func

    def policy_gradient(self):
        pass

    def fit(self, epochs: int = 10_000):
        best = 0
        last_save = 0
        with open(self.out.with_suffix(".log"), "w") as log:
            for epoch in range(epochs):
                print("\n----------\nEPOCH %d\n----------" % epoch)
                self.policy_gradient()
                seqs = self.agent.sample(self.n_samples)
                ix = src.drugexr.utils.tensor_ops.unique(seqs)
                smiles = [self.agent.voc.decode(s) for s in seqs[ix]]
                scores = self.env(smiles, is_smiles=True)

                desire = scores.DESIRE.sum() / self.n_samples
                score = scores[self.env.keys].values.mean()
                valid = scores.VALID.mean()

                if best <= score:
                    torch.save(self.agent.state_dict(), self.out.with_suffix(".pkg"))
                    best = score
                    last_save = epoch

                print(
                    "Epoch: %d average: %.4f valid: %.4f unique: %.4f"
                    % (epoch, score, valid, desire),
                    file=log,
                )
                for i, smile in enumerate(smiles):
                    score = "\t".join(["%0.3f" % s for s in scores.values[i]])
                    print("%s\t%s" % (score, smile), file=log)
                if epoch - last_save > 100:
                    break
            for param_group in self.agent.optim.param_groups:
                param_group["lr"] *= 1 - 0.01
