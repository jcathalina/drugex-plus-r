from typing import List

import numpy as np
import rdkit.Chem
import torch
from torch.utils.data.dataset import Dataset, random_split

from src.drugexr.data.preprocess import logger
# TODO: Export to test module and put large objects into files
from src.drugexr.models.predictor import Predictor

SAMPLE_SMILES = [
    "CC(C)(N)CP(=O)(O)CCCCc1ccc(C#Cc2ccccc2F)cc1",
    "c1ccc2c(CC3CNCCC3O)cccc2c1",
    "c1cc2cc(-c3cnc4ccccc4c3)c[nH]c2n2",
    "CSc1cc2c(c(C#N)c1)C(C)N(C)C2",
    "Cc1onc(-c2ccccc2)c1C(=O)Nc1cccc(N2C(=O)CSC2c2cc(F)ccc2F)n1",
    "Cn1c(=O)c2c(nc(C=Cc3cccc(Br)c3)nn2C)n(C)c1=O",
    "NC(=S)c1cccc(OCCCc2ccccc2OCc2ccccc2)c1",
    "O=C(Cc1ccc(-c2noc(C(F)(F)F)n2)cc1)NC(Cc1ccc(OCc2ccnc3ccccc23)cc1)C(=O)O",
    "CC(C)=CCN1CCN(c2cccc3nc(-c4ccc(OC5CCN(C)CC5)cc4)c(=N)oc23)CC1",
    "Nc1nc2ccccc2[nH]1",
    "CCNC1CN(c2ncnc3[nH]ccc23)C1",
    "O=C(Nc1ccc(CN2CCCC2)cc1)c1cc(-c2ccccc2)nc2ccsc12",
    "COc1cccc2c1nc(N)n1nc(-c3ccco3)nn12",
    "Cn1cc2c(nc(NCc3ccc(CN(C)S(C)(=O)=O)cc3)n3nc(-c4ccco4)nc23)n1",
    "Nc1nc2ccccc2n2nc(-c3ccco3)nn12",
    "Nc1nc2ccccc2n2c(SCC(O)CO)nnc1-2",
    "CCCn1c(=O)c2[nH]c(-c3ccc(OC(C)(C)C(=O)Nc4cc(N(C)C)no3)cc3)[nH]c2n(CCC)c1=O",
    "CC1(C)Oc2ccc(F)cc2C2(COC(N)=N2)C12COC2",
    "O=C(CN1CCc2ccccc2C1)N1CCN(c2ccc(Cl)cc2)C(=N)C1",
    "COc1cccc(-c2cc(OCCCC(C)N3CCC(C)c4ccccc4C3)ncn2)c1",
    "COc1ccc(-c2cn3nc(-c4ccco4)nn3c2-c2ccccc2)cc1",
    "CCn1c(=O)c2nc(-c3cccnc3)[nH]c2n(CCC)c1=O",
    "CNC(=O)c1cc(COc2nc(N)c3ncn(C4OC(CO)C(O)C(O)C4O)c3n2)ccn1",
    "COc1cc(-c2n[nH]c3c2C(=O)c2ccccc2-3)cc(OC)c1Cl",
    "CCCn1c(=O)c2nc(-c3cnn(Cc4nc(-c5ccc(C)cc5)co4)n3-c3ccc(C)cc3)[nH]c(=O)n2-c2ccccc21",
    "CCCCCCN(C(=O)Cc1ccc(OC)c(OC)c1)C1CCN(c2ccc(C(=O)O)cn2)CC1",
    "Cc1cc(C)n(-c2ccc(N(C)C)cc2)n1",
    "CC1CN(CC2CN(C(C)=O)CC2c2ccccc2)CCN1c1ncc(C(N)=O)cn1",
    "ON=C(c1cccc(-c2ccc(F)cc2)c1)c1ncco1",
    "Cc1ccc2c(N3CCN(CCCN4C(=O)c5ccccc5Sc5c4ccccc4Br)CC3)c(Cl)cnc2c1",
    "CCOC(=O)C1=C(CN2CCOCC2)NC(n2nnc(C)c2N=Nc2ccccc2)N=C1",
    "COCc1cn2c3c(ccc2s1)C(=O)NCc1ccncc1-3",
    "NC1CN(c2ccccc2F)CC1c1ncc[nH]1",
    "CC(Nc1nc(N)c(OCCO)c2nc[nH]c12)C(O)c1ccccc1",
    "CC(C)CN1CC2CC(N(C)Cc3nc4ccccc4s3)CC2C1=O",
    "CCNC(=O)C1OC(n2cnc3c(NC)nc(C#CCC(=O)Nc4ccccc4)nn3)ccC2=O)cc1",
    "Nc1nc(C(=O)NCc2ccncc2)cn2nc(-c3ccco3)nn12",
    "CCCn1c(=O)n(C)c(=O)c2[nH]c(O)c3c2C(=O)c2ccccc2n3C1C",
    "CCN(C(C)C)C1CCC(C(=O)Nc2cccnc2)=C(c2ccccc2)C1=O",
    "CC1(c2cc(NC(=O)c3ncc(F)cn3)ccc2F)N=C(N)COC(CF)C1O",
    "NNC(=O)c1cccc(Cc2ccc3c(c2)nc(-c2ccccc2)n3Cc2ccccc2)c1",
    "CNCc1cc(-c2sccc2C)nc(N)n1",
    "CNCCC(c1nc2cc(Cl)ccc2n1C)n1nc(-c2ccccc2)nc2c(Cl)cccc12",
    "CCNc1nc(-n2CCC3(CCN(CCO)CC3)CC2)nc2nc(C)c(C)nc1-2",
    "Cc1nsc(-c2nc(SC)c3ncccn3n2)n1",
    "CC(=O)NC1C(C)OC(n2cnc3c(NC4CCOC4)ncnc32)C(O)C1O",
    "Nc1nc2ccc(Cl)cc2c2nc(-c3ccco3)nn12",
    "COc1ccc(Cn2c(Cl)nc3c(N)nc(-c4ccccc4)nc32)cc1",
    "CC(C)Cn1c(=O)n(C)c(=O)c2[nH]c(-c3cnn(Cc4ccc(F)cc4)c3)cnc21",
    "Cc1nc2ccccc2n1CC(=O)NCc1ncc(C(F)(F)F)cc1Cl",
    "CCCn1c(=O)c2[nH]c(-c3ccc(OCC(=O)Nc4ccccn4)cc3)cc2n(CCC)c1=O",
    "CC=CC(=O)Nc1cc(Nc2ccc(F)cc2Cl)nc(NC(=O)c2ccccc2Cl)c1",
    "CC(C)Cn1ncc2c1CC(C)(C)CC2=NNc1ccnc2cc(Cl)ccc12",
    "Nc1nc(SC2CCCCC2)nc2c1ncn2C1OC(O)C(O)C2O)c(Cc3ccccc3)nc21",
    "OCC1OC(n2cnc3c(NC4CCCC4)ccnc32)C(O)C1O",
    "CC(C)C(NC(=O)C1(N)CCN(C(c2ccccc2)c2ccccc2)CC1)C(=O)C(F)(F)F",
    "Nc1nc2cc3c(cc2s1)CCN(Cc1ccncc1)CC3",
    "N#Cc1cccc2c1C=Cc1ncccc1-2",
    "CNCCCCC(c1ccccc1)c1ccccc1",
    "Cc1cc2c3c(nc(C)nn3c1C)CCC=C2c1ccc(F)cc1",
    "Nc1nc(SCCc2ccccc2)nc2nc(-c3ccco3)nn12",
    "Cc1cccc(Nc2sc3ccc(Cl)cc3c2C2CCN(C(C)C)CC2)c1C",
    "Clc1ccc(Nc2nc3ccccc3c3[nH]c(1)c(=N)n23)cc1Cl",
    "CCn1c(=O)n(CCC(C)C)c2cccnc21",
    "CCCn1c(=O)c2nc(-c3cnn(Cc4nc(-c5ccc(F)cc5)no4)c3)[nH]cc2n(CCC)c1=O",
    "N#Cc1cnccc1-c1cccc2cc(-c3cnn4ccccc34)on12",
    "CON=C(c1ccnn1C)c1nc(-c2ccncc2)no1",
    "COCCCn1c(=O)c2nc(-c3ccccc3)[nH]c2n(CC)c1=O",
    "CC1=C(C(=O)c2ccccc2)C(C)(C)c2nc(CCCN3C(=S)NCC3)c[nH]c21",
    "COc1cc(OC)cc(C2CC3CN(CCCCSc4nc5ccccc5[nH]4)nn3)CC2CN1",
    "Nc1scc(CN2CCN(c3ccccc3F)CC2)c1C(=O)c1ccccc1",
    "C=C1CCC2C(C)(CO)CCCC2(C)C1CCC1(C)C2CC=C3C(C)CCCC3(C)C1CC21CCC(NC(C)=O)CC1",
    "Cn1cc2cc3c(nc(O)n3nc2-c2ccccc2)n1C1CCCCC1",
    "Fc1cccc(C2(N3CCCC3)CC2)c1",
    "CCCn1c(=O)c2[nH]c(-c3cnn(CC#CCCCC(=O)NO)n3)n(CC)n2n(CC)c1=O",
    "N#Cc1cnc(-c2ccccc2)cn1",
    "CC(=O)NC1COC(n2cnc3c(NC4CCCCC4)ncnc32)C(O)C1O",
    "COc1cccc2c1nc(N)n1nc(-c3ccco3)nn12",
    "CNC(=O)C1c2ccccc2C(=O)N2CC(c3ccoc3)C3(CCCN3C)CN21",
    "Cn1nccc1-c1cc(NC(=O)Nc2ccc(F)cc2)no1",
    "CS(=O)(=O)NCCNc1nc(NN=CC=C2CCCN3C(=O)CC3c2nnn1CCc1ccccc1)n1cccn1",
    "Nc1scc(CNc2cccn3ccccc23)c1-c1ccc2ccccc2n1",
    "Nc1nc2ccccc2n2c(-c3cc(Cl)ccc3Cl)nnc12",
    "CCCn1c(=O)c2nc(-c3cnn(Cc4noc(-c5ccc6c(c5)OCCO6)n3)cc3)[nH]c2n(CCC)c1=O",
    "Nc1nc(C(=O)NCc2nccs2)cc2ncn(Cc3ccccc3F)n12",
    "CC1CCC(C(C)C)C(OC(=O)N2CCN(C(=O)Cc3ccccc3)CC2)=C1",
    "CN1CCCC1Cn1nc(C(=O)NCc2ccccc2)c(OCc2ccccc2)c2ccccc12",
    "O=C1N(CCN2Cc3ccccc3C2)Cc2nc(-c3ccccc3Cl)n(C3CC4CCCC3C4)cc21",
    "COc1ccc(-c2noc(C3COCCO3)n2)cc1OCCN1CCCCC1",
    "Cn1cnc2c(Oc3ccc(C(=O)NC4CC4)c(F)c3)cc(Oc3ccccc3F)ccc21",
    "CCNC(=O)C1OC(n2cnc3c(NC4CCN(C(C)=O)CC4)ncnc32)C(O)C1O",
    "CCCn1c(NC2CCN(C)CC2)nc2cc(C=CC(=O)NO)ccc21",
    "NCCCc1c(I)cccc1Oc1ccccc1",
    "Nc1scc(CN2CCN(c3ccccc3)CC2)c1C(=O)c1ccccc1",
    "COc1cc(NS(C)(=O)=O)ccc1-c1cnc(NCc2ccccc2)nn1",
    "CCn1c(-c2nonc2N)nc2ccccc21",
    "Nc1nc2cc(-c3ccccc3)nnc2[nH]1",
    "CCCn1c(=O)c2[nH]c(-c3ccc(OCC(=O)Nc4ccccc4)cc3)cc2n(C)c1=O",
    "CCCCn1cc2c(nc(NC(=O)Nc3ccc(Br)cc3)n3nc(-c4ccco4)nn23)n1",
    "CCCn1c(=O)c2[nH]c(-c3ccc(OCC(=O)Nc4ccc(OC)c(F)c4)cc3)cc2n(CCC)c1=O",
]


def unique(arr):
    """Finds unique rows in arr and return their indices"""
    if type(arr) == torch.Tensor:
        arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(
        np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    )
    _, idxs = np.unique(arr_, return_index=True)
    idxs = np.sort(idxs)
    if type(arr) == torch.Tensor:
        idxs = torch.LongTensor(idxs).to(arr.get_device())
    return idxs


def canonicalize_smiles_list(smiles: List[str]) -> List[str]:
    canon_smiles = []
    for smi in smiles:
        try:
            canon_smi = rdkit.Chem.CanonSmiles(smi=smi)
            canon_smiles.append(canon_smi)
        except Exception as e:
            logger.warn(f"{smi} is not a valid Molecule: {e}, skipping...")
            continue
    return canon_smiles


def random_split_frac(dataset: Dataset, train_frac: float = 0.9, val_frac: float = 0.1):
    """
    Helper wrapper function around PyTorch's random_split method that allows you to pass
    fractions instead of integers.
    """
    if train_frac + val_frac != 1:
        raise ValueError("The fractions have to add up to 1.")

    dataset_size = len(dataset)

    len_1 = np.floor(train_frac * dataset_size)
    len_2 = dataset_size - len_1
    return random_split(dataset=dataset, lengths=[len_1, len_2])


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    # artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    # print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def test_canonicalize_smiles_list():
    x = canonicalize_smiles_list(SAMPLE_SMILES)
    print(x)


def test_unique():
    sample_smiles_arr = np.array([[s] for s in SAMPLE_SMILES])
    unique_indices = unique(arr=sample_smiles_arr)
    x = sample_smiles_arr[unique_indices]
    print(x.shape)


def test_predictor_calcs():
    x = Predictor.calc_physchem(mols=[])
    print(x)


if __name__ == "__main__":
    test_unique()
