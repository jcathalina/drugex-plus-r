#!/usr/bin/env python

# Based on the code found in Pytorch-Lightning's setup.py (link below)
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/setup.py

import os
from importlib.util import module_from_spec, spec_from_file_location

from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)


def _load_py_module(fname, pkg="src/drugexr"):
    spec = spec_from_file_location(
        os.path.join(pkg, fname), os.path.join(_PATH_ROOT, pkg, fname)
    )
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


about = _load_py_module("__about__.py")


setup(
    name="drugexr",
    version=about.__version__,
    description=about.__docs__,
    author=about.__author__,
    author_email=about.__author_email__,
    url=about.__homepage__,
    license=about.__license__,
    download_url="https://github.com/naisuu/drugex-plus-r",
    python_requires=">=3.7",
    platforms=["Linux", "Windows"],
    packages=find_packages(where="src", include=["drugexr"]),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "download_raw_chembl=drugexr.tools.download_raw_chembl:download_chembl_data",
        ],
    },
    keywords=[
        "deep learning",
        "reinforcement learning",
        "CASP",
        "drug design",
        "retrosynthesis",
    ],
    classifiers=[
        "Programming Language :: Python :: 3 :: Only"
        "Programming Language :: Python :: 3.7"
    ],
)
