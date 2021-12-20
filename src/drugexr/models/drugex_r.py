from enum import Enum

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from drugexr.config.constants import DEVICE
from drugexr.data_structs.environment import Environment
from drugexr.models.generator import Generator
from drugexr.utils import tensor_ops


class MeanFn(Enum):
    GEOMETRIC = 1
    ARITHMETIC = 2


class RewardScheme(Enum):
    PARETO_FRONT = 1
    WEIGHTED_SUM = 2


class DrugExR(pl.LightningModule):
    """ """

    def __init__(
        self,
        agent: Generator,
        prior: Generator,
        xover_net: Generator,
        mutator_net: Generator,
        environment: Environment,
        mean_fn: MeanFn = MeanFn.GEOMETRIC,
        scheme: RewardScheme = RewardScheme.PARETO_FRONT,
        epsilon: float = 1e-3,
        batch_size: int = 512,
        n_samples: int = 512,
        replay: int = 10,
    ):
        super.__init__()
        self.agent = agent
        self.prior = prior
        self.xover_net = xover_net
        self.mutator_net = mutator_net
        self.environment = environment
        self.mean_fn = mean_fn
        self.scheme = scheme
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.replay = replay

    def policy_gradient(self):
        """ """
        sequences = [
            self.agent.evolve(
                batch_size=self.batch_size,
                epsilon=self.epsilon,
                crover=self.xover_net,
                mutate=self.mutator_net,
            )
            for _ in range(self.replay)
        ]

        sequences = torch.cat(sequences, dim=0)
        smiles = np.array([self.agent.voc.decode(s) for s in sequences])
        indices = tensor_ops.unique(np.array([[s] for s in smiles]))
        smiles = smiles[indices]
        sequences = sequences[torch.LongTensor(indices).to(DEVICE)]

        scores = self.environment.calc_reward(smiles=smiles, scheme=self.scheme)
        dataset = TensorDataset(sequences, torch.Tensor(scores).to(DEVICE))
        loader = DataLoader(dataset=dataset, batch_size=self.n_samples, shuffle=True)

        self.agent.policy_gradient_loss(loader=loader)
