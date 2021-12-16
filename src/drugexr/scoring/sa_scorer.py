#
# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# several small modifications to the original paper are included
# particularly slightly different formula for macrocyclic penalty
# and taking into account also molecule symmetry (fingerprint density)
#
# for a set of 10k diverse molecules the agreement between the original method
# as implemented in PipelinePilot and this implementation is r2 = 0.97
#
# Peter Ertl & Greg Landrum, september 2013


import gzip
import math
import os.path as op
import pickle
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdchem, rdMolDescriptors

_fscores: Optional[Dict] = None


def read_fragment_scores(name: str = "fpscores") -> None:

    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open(f"{name}.pkl.gz"))
    out_dict = {}
    for i in data:
        for j in range(1, len(i)):
            out_dict[i[j]] = float(i[0])
    _fscores = out_dict


def num_bridgeheads_and_spiro(mol: rdchem.Mol) -> Tuple[int, int]:
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return n_bridgehead, n_spiro


def calculate_score(mol: rdchem.Mol, circ_fp_radius: int = 2) -> float:
    if _fscores is None:
        read_fragment_scores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(mol=mol, radius=circ_fp_radius)
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bit_id, v in fps.items():
        nf += v
        sfp = bit_id
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    n_atoms = mol.GetNumAtoms()
    n_chiral_centers = len(Chem.FindMolChiralCenters(mol=mol, includeUnassigned=True))
    ring_info = mol.GetRingInfo()
    n_bridgeheads, n_spiro = num_bridgeheads_and_spiro(mol=mol)

    n_macrocycles = 0
    for x in ring_info.AtomRings():
        if len(x) > 8:
            n_macrocycles += 1

    size_penalty = n_atoms ** 1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral_centers + 1)
    spiro_penalty = math.log10(n_spiro + 1)
    bridge_penalty = math.log10(n_bridgeheads + 1)
    macrocycle_penalty = 0.0
    # ---------------------------------------
    # This differs from the paper, which defines:
    # macrocyclePenalty = math.log10(n_macrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if n_macrocycles > 0:
        macrocycle_penalty = math.log10(2)

    score2 = (
        0.0
        - size_penalty
        - stereo_penalty
        - spiro_penalty
        - bridge_penalty
        - macrocycle_penalty
    )

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthesise
    score3 = 0.0
    if n_atoms > len(fps):
        score3 = math.log(float(n_atoms) / len(fps)) * 0.5

    sa_score = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min_ = -4.0
    max_ = 2.5
    sa_score = 11.0 - (sa_score - min_ + 1) / (max_ - min_) * 9.0
    # smooth the 10-end
    if sa_score > 8.0:
        sa_score = 8.0 + math.log(sa_score + 1.0 - 9.0)
    if sa_score > 10.0:
        sa_score = 10.0
    elif sa_score < 1.0:
        sa_score = 1.0

    return sa_score


def process_mols(mols: List[rdchem.Mol]) -> None:
    print("smiles\tName\tsa_score")
    for mol in mols:
        if mol is None:
            continue

        score = calculate_score(mol=mol)

        smiles = Chem.MolToSmiles(mol)
        print(smiles + "\t" + mol.GetProp("_Name") + "\t%3f" % score)
