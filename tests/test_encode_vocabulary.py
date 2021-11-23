import os

from pyprojroot import here

from src.drugexr.data_structs.vocabulary import Vocabulary


def count_lines(filepath: os.PathLike) -> int:
    with open(filepath, "r") as f:
        count = len(f.readlines())
    return count


def test_vocabulary_size_equals_lines_plus_controls():
    root = here(project_files=[".here"])
    vocabulary_path = root / "data/processed/vocabulary.txt"

    controls = 2
    expected_count = count_lines(vocabulary_path) + controls
    voc = Vocabulary(vocabulary_path=vocabulary_path)
    assert voc.size == expected_count
