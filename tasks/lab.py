from invoke import task

from . import config


@task(default=True)
def init(ctx, env_name=config.env_name, port=None):
    """
    Initiates jupyter lab in this project using
    the given environment and port

    Parameters
    ----------
    env_name : str
        The name of the environment used for this project
    port : str
        The port to host the notebook on
    """
    args = []
    if port is not None:
        args.append(f"--port={port}")

    cmd = [f"conda activate {env_name}",
           f"jupyter lab" + " ".join(args)
    ]
    
    ctx.run(" && ".join(cmd))

    
