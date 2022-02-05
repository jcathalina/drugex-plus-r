from typing import List

import numpy as np
import torch

from drugexr.models.generator import Generator
from drugexr.utils import tensor_ops


class Agent:
    """"""
    def __init__(self, net: Generator):
        self.net = net

    def __call__(self, n_samples: int = 128, device: str = "cuda", *args, **kwargs) -> List[str]:
        """Using the given network, decide what action to carry.
        Args:
            state: current state of the environment
            device: device used for current batch
        Returns:
            action
        """
        sequences = self.net.sample(batch_size=n_samples)
        smiles = [self.net.voc.decode(seq) for seq in sequences]
        smiles = np.array(tensor_ops.unique(np.array([[smi] for smi in smiles])))
        indices = tensor_ops.unique(np.array([[smi] for smi in smiles]))
        smiles = smiles[indices]

        return smiles
