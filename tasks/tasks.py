from invoke import task


@task
def clean(c):
    """
    Removes already existing documentation files
    """
    c.run("rm -rf docs/_build")


@task(pre=[clean])  # clean is run as a pre-task to build, as we always want this to happen first.
def build(c):
    """
    Automatically build the documentation for DrugEx MINUS.
    """
    c.run("sphinx-build docs docs/_build")
