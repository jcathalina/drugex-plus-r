import os
from typing import Optional


class Vocabulary:
    """"""

    def __init__(self, init_from_file: Optional[os.PathLike] = None, max_len: int =100):
        """"""
        self.control = ['EOS', 'GO']
        self.words = [] + self.control
        if init_from_file:
            self.words += self.init_from_file(init_from_file)
        self.size = len(self.words)
        self.tk2ix = dict(zip(self.words, range(len(self.words))))
        self.ix2tk = {v: k for k, v in self.tk2ix.items()}
        self.max_len = max_len

    def init_from_file(self, init_from_file):
        pass