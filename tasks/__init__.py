from invoke import Collection

from . import env, config, lab

ns = Collection(env, lab)
ns.configure({"run": {"shell": config.shell, "pty": config.pty}})