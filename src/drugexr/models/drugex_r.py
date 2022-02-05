import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm

from drugexr.config.constants import DEVICE
from drugexr.data_structs.environment import Environment
from drugexr.models.generator import Generator
from drugexr.utils import tensor_ops
from drugexr.utils.enums import RewardScheme, MeanFn


class DrugExR:
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
        batch_size: int = 64,
        n_samples: int = 128,
        replay: int = 10,
    ):
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

    def policy_gradient(self) -> None:
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


        sequences = torch.cat(sequences, dim=0).to(DEVICE)
        smiles = np.array([self.agent.voc.decode(s) for s in sequences])
        indices = tensor_ops.unique(np.array([[smi] for smi in smiles]))
        smiles = smiles[indices]
        sequences = sequences[torch.tensor(indices, dtype=torch.long, device=DEVICE)]

        scores = self.environment.calc_reward(smiles=smiles, scheme=self.scheme)
        dataset = TensorDataset(sequences, torch.tensor(scores, device=DEVICE))
        loader = DataLoader(dataset=dataset, batch_size=self.n_samples, shuffle=True)

        self.agent.policy_gradient_loss(loader=loader)
        # FIXME: Things break here because of CPU/GPU discrepancy, debug.

    def fit(self, output_path: Path, epochs: int = 10_000, interval: int = 250):
        """ """
        best_score = 0
        last_smiles = []
        last_scores = []
        interval = interval
        last_save = -1

        for epoch in tqdm(range(epochs), desc="Main RL Loop"):
            self.policy_gradient()
            sequences = self.agent.sample(self.n_samples)
            smiles = [self.agent.voc.decode(seq) for seq in sequences]
            smiles = np.array(tensor_ops.unique(np.array([[smi] for smi in smiles])))
            indices = tensor_ops.unique(np.array([[smi] for smi in smiles]))
            smiles = smiles[indices]
            scores = self.environment.__call__(mols=smiles, is_smiles=True)

            desire = scores["DESIRE"].sum() / self.n_samples
            if self.mean_fn == MeanFn.ARITHMETIC:
                score = (
                    scores[self.environment.keys].values.sum()
                    / self.n_samples
                    / len(self.environment.keys)
                )
            elif self.mean_fn == MeanFn.GEOMETRIC:
                score = scores[self.environment.keys].values.prod(axis=1) ** (
                    1.0 / len(self.environment.keys)
                )
                score = score.sum() / self.n_samples
            else:
                raise ValueError(f"Selected mean function {self.mean_fn} does not exist.")

            valid = scores["VALID"].sum() / self.n_samples

            logging.info(
                f"Epoch: {epoch+1} average: {score:.4} valid: {valid:.4} desire: {desire:.4}"
            )

            if best_score < score:
                torch.save(self.agent.state_dict(), output_path.with_suffix(".pkg"))
                best_score = score
                last_smiles = smiles
                last_scores = scores
                last_save = epoch

            if epoch % interval == 0 and epoch != 0:
                for i, smile in enumerate(last_smiles):
                    score = "\t".join(["%.3f" % s for s in last_scores.values[i]])
                    logging.info(f"{score}\t{smile}")
                self.agent.load_state_dict(torch.load(output_path.with_suffix(".pkg")))
                self.xover_net.load_state_dict(
                    torch.load(output_path.with_suffix(".pkg"))
                )
            if epoch - last_save > interval:
                break
