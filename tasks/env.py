from os import name
from invoke import task

from . import config as cfg

@task
def clean(ctx, env_name=cfg.env_name):
    """
    Deletes the conda environment

    Parameters
    ----------
    env_name : str
        The name of the environment that will be deleted
    """
    if cfg.is_root_env(env_name):
        print(f"Could not remove environment: {env_name}")
    else:
        ctx.run(f"conda deactivate && conda remove -n {env_name} --all")

@task
def build(ctx, env_name=cfg.env_name):
    """
    Builds the environment necessary to run this project
    """
    cmd = [f"conda activate {env_name}"]
    len_min = len(cmd)

    #  Install Jupyterlab extensions
    if cfg.jupyterlab_exts:
        cmd.append("jupyter lab clean")
        cmd.extend(
            f"jupyter labextension install {ext} --no-build"
            for ext in cfg.jupyterlab_exts
        )
        
    if len(cmd) > len_min:
        print("Running cmd: " + " && ".join(cmd))
        ctx.run(" && ".join(cmd))
    
    ctx.run(f"conda activate {env_name} && poetry install")

@task
def init(
    ctx,
    env_name=cfg.env_name,
    env_file=cfg.env_file,
    clean_env=False
):
    """
    Creates or reinstalls the environment for this project

    Parameters
    ----------
    env_name : str
        The name of the environment that will be deleted
    env_file : str
        The name (filepath) of the YAML file describing the environment
    clean_env : bool
        False by default, if set to true, the environment is first removed if it already exists and then rebuilt
    """
    if cfg.is_root_env(env_name):
        ctx.run(f"conda env update -n {env_name} -f {env_file}")
    elif clean_env:
        clean(ctx, env_name)
    ctx.run(f"conda env create -n {env_name} -f {env_file}")
    

    build(ctx, env_name)


@task
def aaa(ctx, env_name=cfg.env_name):
    ctx.run(f"activate {env_name}")