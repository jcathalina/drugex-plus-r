import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from drugexr.config.constants import DEVICE
from drugexr.models.pg_learner import PGLearner
from drugexr.utils import tensor_ops


class DrugExV2(PGLearner):
    """DrugEx algorithm (version 2.0)
    Reference: Liu, X., Ye, K., van Vlijmen, H.W.T. et al. DrugEx v2: De Novo Design of Drug Molecule by
               Pareto-based Multi-Objective Reinforcement Learning in Polypharmacology.
               J Cheminform (2021). https://doi.org/10.1186/s13321-019-0355-6
    Arguments:
        agent (models.Generator): The agent network which is constructed by deep learning model
                                   and generates the desired molecules.
        env (utils.Env): The environment which provides the reward and judge
                                 if the genrated molecule is valid and desired.
        prior (models.Generator): The pre-trained network which is constructed by deep learning model
                                   and ensure the agent to explore the approriate chemical space.
    """

    def __init__(
        self,
        agent,
        env,
        prior=None,
        crover=None,
        mean_func="geometric",
        memory=None,
    ):
        super(DrugExV2, self).__init__(
            agent, env, prior, mean_func=mean_func, memory=memory
        )
        self.crover = crover

    def policy_gradient(self, crover=None, memory=None, epsilon=None):
        seqs = []
        start = time.time()
        for _ in range(self.replay):
            seq = self.agent.evolve(
                self.batch_size, epsilon=epsilon, crover=crover, mutate=self.prior
            )
            seqs.append(seq)
        t1 = time.time()
        seqs = torch.cat(seqs, dim=0)
        if memory is not None:
            mems = [memory, seqs]
            seqs = torch.cat(mems)
        smiles = np.array([self.agent.voc.decode(s) for s in seqs])
        # smiles = np.array(utils.canonicalize_list(smiles))
        ix = tensor_ops.unique(np.array([[s] for s in smiles]))
        smiles = smiles[ix]
        seqs = seqs[torch.LongTensor(ix).to(DEVICE)]

        scores = self.env.calc_reward(smiles, self.scheme)
        if memory is not None:
            scores[: len(memory), 0] = 1
            ix = scores[:, 0].argsort()[-self.batch_size * 4 :]
            seqs, scores = seqs[ix, :], scores[ix, :]
        t2 = time.time()
        ds = TensorDataset(seqs, torch.Tensor(scores).to(DEVICE))
        loader = DataLoader(ds, batch_size=self.n_samples, shuffle=True)

        self.agent.PGLoss(loader)
        t3 = time.time()
        print(t1 - start, t2 - t1, t3 - t2)

    def fit(self, epochs: int = 10_000):
        best = 0
        log = open(self.out.with_suffix(".log"), "w")
        last_smiles = []
        last_scores = []
        interval = 250
        last_save = -1

        for epoch in range(epochs):
            print("\n----------\nEPOCH %d\n----------" % epoch)
            if epoch < interval and self.memory is not None:
                self.policy_gradient(crover=None, memory=self.memory, epsilon=1e-1)
            else:
                self.policy_gradient(crover=self.crover, epsilon=self.epsilon)
            seqs = self.agent.sample(self.n_samples)
            smiles = [self.agent.voc.decode(s) for s in seqs]

            # TODO: This section can probably be optimized --> less conversions to np arrays?
            smiles = np.array(
                tensor_ops.canonicalize_smiles_list(smiles)
            )  # TODO: Confirm if this step is critical
            ix = tensor_ops.unique(np.array([[s] for s in smiles]))
            smiles = smiles[ix]
            scores = self.env(smiles, is_smiles=True)

            desire = scores.DESIRE.sum() / self.n_samples
            if self.mean_func == "arithmetic":
                score = (
                    scores[self.env.keys].values.sum()
                    / self.n_samples
                    / len(self.env.keys)
                )
            else:
                score = scores[self.env.keys].values.prod(axis=1) ** (
                    1.0 / len(self.env.keys)
                )
                score = score.sum() / self.n_samples
            valid = scores.VALID.sum() / self.n_samples

            print(
                "Epoch: %d average: %.4f valid: %.4f unique: %.4f"
                % (epoch, score, valid, desire),
                file=log,
            )
            if best < score:
                torch.save(self.agent.state_dict(), self.out.with_suffix(".pkg"))
                best = score
                last_smiles = smiles
                last_scores = scores
                last_save = epoch

            if epoch % interval == 0 and epoch != 0:
                for i, smile in enumerate(last_smiles):
                    score = "\t".join(["%.3f" % s for s in last_scores.values[i]])
                    print("%s\t%s" % (score, smile), file=log)
                self.agent.load_state_dict(torch.load(self.out.with_suffix(".pkg")))
                self.crover.load_state_dict(torch.load(self.out.with_suffix(".pkg")))
            if epoch - last_save > interval:
                break
        log.close()
