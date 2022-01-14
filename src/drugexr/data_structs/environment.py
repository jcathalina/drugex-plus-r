from typing import List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem

from drugexr.models.predictor import Predictor
from drugexr.utils.enums import RewardScheme
from drugexr.utils.fingerprints import get_fingerprint
from drugexr.utils.sorting import nsgaii_sort, similarity_sort


class Environment:
    def __init__(self,
                 objs: List,
                 mods: List,
                 keys: List[str],
                 ths: Optional[List[float]] = None):
        """
        Initialized methods for the construction of environment.
        Args:
            objs (List[Any]): a list of objectives.
            mods (List[Any]): a list of modifiers, and its length
                equals the size of objs.
            keys (List[str]): a list of strings as the names of objectives,
                and its length equals the size of objs.
            ths (Optional[List[float]]): a list of float value, and ts length equals the size of objs.
        """
        self.objs = objs
        self.mods = mods
        self.ths = ths if ths is not None else [0.99] * len(keys)
        self.keys = keys

    def __call__(
        self, mols: List[str] = None, is_smiles: bool = False, is_modified: bool = True
    ):
        """
        Calculate the scores of all objectives for all of samples
        Args:
            mols (List): a list of molecules
            is_smiles (bool): if True, the type of element in mols should be SMILES sequence, otherwise
                it should be the Chem.Mol
            is_modified (bool): if True, the function of modifiers will work, otherwise
                the modifiers will ignore.
        Returns:
            preds (DataFrame): The scores of all objectives for all of samples which also includes validity
                and desirability for each SMILES.
        """
        preds = {}
        fps = None
        if is_smiles:
            mols = [Chem.MolFromSmiles(s) for s in mols]
        for i, key in enumerate(self.keys):
            # TODO: should probably be using is instance of instead...
            if type(self.objs[i]) == Predictor:
                if fps is None:
                    fps = Predictor.calc_fp(mols)
                score = self.objs[i](fps)
            else:
                score = self.objs[i](mols)
            if is_modified and self.mods[i] is not None:
                score = self.mods[i](score)
            preds[key] = score
        preds = pd.DataFrame(preds)
        undesire = preds < self.ths  # ^ self.objs.on
        preds["DESIRE"] = (undesire.sum(axis=1) == 0).astype(int)
        preds["VALID"] = [0 if mol is None else 1 for mol in mols]
        preds[preds.VALID == 0] = 0
        return preds

    @classmethod
    def calc_fps(cls, mols, fp_type="ECFP6"):
        fps = []
        for i, mol in enumerate(mols):
            try:
                fps.append(get_fingerprint(mol, fp_type))
            except:
                fps.append(None)
        return fps

    def calc_reward(self, smiles: List[str], scheme: RewardScheme = RewardScheme.WEIGHTED_SUM):
        """
        Calculate the single value as the reward for each molecule used for reinforcement learning
        Args:
            smiles (List[str]):  a list of SMILES-based molecules
            scheme (str): the label of different rewarding schemes, including
                'WS': weighted sum, 'PR': Pareto ranking with Tanimoto distance,
                and 'CD': Pareto ranking with crowding distance.
        Returns:
            rewards (np.ndarray): n-d array in which the element is the reward for each molecule, and
                n is the number of array which equals to the size of smiles.
        """
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        preds = self(mols)
        desire = preds.DESIRE.sum()
        undesire = len(preds) - desire
        preds = preds[self.keys].values

        if scheme == RewardScheme.PARETO_FRONT:
            fps = self.calc_fps(mols)
            rewards = np.zeros((len(smiles), 1))
            ranks = similarity_sort(preds, fps, is_gpu=True)
            score = (np.arange(undesire) / undesire / 2).tolist() + (
                np.arange(desire) / desire / 2 + 0.5
            ).tolist()
            rewards[ranks, 0] = score
        elif scheme == RewardScheme.CROWDING_DISTANCE:
            rewards = np.zeros((len(smiles), 1))
            ranks = nsgaii_sort(preds, is_gpu=True)
            rewards[ranks, 0] = np.arange(len(preds)) / len(preds)
        elif scheme == RewardScheme.WEIGHTED_SUM:
            weight = ((preds < self.ths).mean(axis=0, keepdims=True) + 0.01) / (
                (preds >= self.ths).mean(axis=0, keepdims=True) + 0.01
            )
            weight = weight / weight.sum()
            rewards = preds.dot(weight.T)
        else:
            raise ValueError(f"Selected weight scheme {scheme} does not exist.")

        return rewards
