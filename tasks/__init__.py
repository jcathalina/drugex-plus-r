from invoke import Collection

from . import env, config, jupyter

ns = Collection(env, jupyter)
ns.configure({"run": {"shell": config.shell, "pty": config.pty}})