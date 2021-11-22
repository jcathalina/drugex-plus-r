from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def clean_mol(smiles: str, is_isomeric: bool = False) -> str:
    """
    Args:
        smiles (str):
        is_isomeric (bool): Determines
    Returns:
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = rdMolStandardize.ChargeParent(mol)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=is_isomeric)
    else:
        return ""
