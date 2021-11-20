# DrugEx+R
[![License](https://img.shields.io/github/license/naisuu/drugex-plus-r)](https://github.com/naisuu/drugex-plus-r/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black) 
[![version](https://img.shields.io/github/v/release/naisuu/drugex-plus-r)](https://github.com/naisuu/drugex-plus-r/releases)

DrugEx+R was created for the purpose of experimenting with the development and implementation of retrosynthesis engines within molecular generators for the goal of de novo drug design; the focus of [my Master's thesis](TODO).
The code in this repository is based on [DrugEx v2](https://github.com/XuhanLiu/DrugEx), released by Xuhan Liu (First Author) and Gerard J.P. van Westen (Correspondent Author) on March 8th, 2021. The same license terms apply for this repository, and can be found in the LICENSE file.

## Getting started
After cloning this repository, make sure you have a conda distribution installed. We recommend [miniforge](https://github.com/conda-forge/miniforge) for licensing reasons, but anaconda/miniconda will work as well.

### Installing the environment
- `conda env create -f environment.yml` or `conda env create -f environment-dev.yml` (for Developers)
- `conda activate dpr`

# Additional information
The paper that accompanies the original DrugEx v2 code can be found [here](https://chemrxiv.org/engage/chemrxiv/article-details/60c75834469df47f67f455b9).
