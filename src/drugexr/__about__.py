import time

_this_year = time.strftime("%Y")
__version__ = "1.0.0"
__author__ = "Julius Cathalina"
__author_email__ = "julius.cathalina@gmail.com"
__license__ = "MIT"
__copyright__ = f"Copyright (c) 2021-{_this_year}, {__author__}."
__homepage__ = "https://github.com/naisuu/drugex-plus-r"
__docs_url__ = "https://github.com/naisuu/drugex-plus-r"
# this has to be simple string, see: https://github.com/pypa/twine/issues/522
__docs__ = "De-novo Drug Design using Deep Reinforcement Learning guided by Retrosynthetic Accessibility scores."

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__docs__",
    "__homepage__",
    "__license__",
    "__version__",
]
