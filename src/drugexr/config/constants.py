from pyprojroot import here


# data directory consts
ROOT_PATH = here(project_files=[".here"])
EXT_DATA_PATH = ROOT_PATH / "data/external"
RAW_DATA_PATH = ROOT_PATH / "data/raw"
PROC_DATA_PATH = ROOT_PATH / "data/processed"
INT_DATA_PATH = ROOT_PATH / "data/intermediate"

# token properties
MIN_TOKEN_LEN = 10
MAX_TOKEN_LEN = 100

# data properties
CHEMBL_26_SIZE = 2_000_000  # Approximately.

