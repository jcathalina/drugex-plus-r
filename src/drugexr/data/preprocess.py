import gzip
import logging
import pathlib
from typing import List

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from drugexr.config.constants import (CHEMBL_26_SIZE, MAX_TOKEN_LEN,
                                      MIN_TOKEN_LEN, PROC_DATA_PATH,
                                      RAW_DATA_PATH, ROOT_PATH)
from drugexr.data_structs.vocabulary import Vocabulary
from drugexr.utils import cleaning

logging.basicConfig(filename=ROOT_PATH / "logs/preprocess.log", level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess(
    raw_data_filepath: pathlib.Path,
    destdir: pathlib.Path,
    is_sdf: bool = False,
    requires_clean: bool = True,
    is_isomeric: bool = False,
):
    """Constructing dataset with SMILES-based molecules, each molecules will be decomposed
    into a series of tokens. In the end, all the tokens will be put into one set as vocabulary.
    Arguments:
        raw_data_filepath (pathlib.Path): The file path of input, either .sdf file or tab-delimited file
        destdir (pathlib.Path): The file path of output
        is_sdf (bool): Designate if the input file is sdf file or not
        requires_clean (bool): If the molecule is required to be clean, the charge metal will be
                removed and only the largest fragment will be kept.
        is_isomeric (bool): If the molecules in the dataset keep conformational information. If not,
                the conformational tokens (e.g. @@, @, \, /) will be removed.
    """
    if is_sdf:
        df = get_mols_from_sdf(is_isomeric, raw_data_filepath)
    else:
        # handle table file
        df = pd.read_table(raw_data_filepath).Smiles.dropna()
    voc = Vocabulary()
    words = set()
    canons = []
    tokens = []
    if requires_clean:
        smiles = set()
        for smile in tqdm(df):
            try:
                smile = cleaning.clean_mol(smile, is_isomeric=is_isomeric)
                smiles.add(Chem.CanonSmiles(smile))
            except Exception as e:
                logger.warning("Parsing Error: ", e)
    else:
        smiles = df.values
    for smile in tqdm(smiles):
        token = voc.tokenize(smile)
        # Only collect the organic molecules
        if {"C", "c"}.isdisjoint(token):
            logger.warning("Non-organic token detected: ", smile)
            continue
        # Remove the metal tokens
        if not {"[Na]", "[Zn]"}.isdisjoint(token):
            logger.warning("Metal token detected: ", smile)
            continue
        # control the minimum and maximum of sequence length.
        if MIN_TOKEN_LEN < len(token) <= MAX_TOKEN_LEN:
            words.update(token)
            canons.append(smile)
            tokens.append(" ".join(token))

    # output the vocabulary file
    with open(destdir / "_voc.txt", "w") as voc_file:
        voc_file.write("\n".join(sorted(words)))

    write_corpus(canon_smiles=canons, destdir=destdir, tokens=tokens)


def write_corpus(canon_smiles, destdir, tokens):
    """Output the dataset file as tab-delimited file"""
    corpus_df = pd.DataFrame()
    corpus_df["Smiles"] = canon_smiles
    corpus_df["Token"] = tokens
    corpus_df.drop_duplicates(subset="Smiles")
    corpus_df.to_csv(path_or_buf=destdir / "_corpus.txt", sep="\t", index=False)


def get_mols_from_sdf(is_isomeric: bool, raw_data_filepath: pathlib.Path) -> List[str]:
    """Handle sdf file with RDkit"""
    inf = gzip.open(raw_data_filepath)
    fsuppl = Chem.ForwardSDMolSupplier(inf)
    smiles = []
    for mol in tqdm(fsuppl, total=CHEMBL_26_SIZE):
        try:
            smiles.append(Chem.MolToSmiles(mol, is_isomeric))
        except Exception as e:
            logger.warning(f"Was not able to convert {mol} to smiles: {e}")
    return smiles


def main():
    preprocess(
        raw_data_filepath=RAW_DATA_PATH / "chembl_26.sdf.gz",
        destdir=PROC_DATA_PATH,
        is_sdf=True,
    )


if __name__ == "__main__":
    main()
