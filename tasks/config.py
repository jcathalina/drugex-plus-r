import os
from shutil import which


def is_root_env(env_name: str) -> bool:
    return env_name in ("root", "base")


env_name = "dpr"
env_file = "env.yml"
env_lock_file = "env-lock.yml"
pkg_name = "drugex-plus-r"

jupyterlab_exts = [
    "@jupyter-widgets/jupyterlab-manager",
    "jupyter-matplotlib",
    "@jupyterlab/toc",
]

shell = which("bash") if os.name != "nt" else which("cmd")
pty = False if os.name == "nt" else True