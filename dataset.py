import torch
from torch.utils.data import Dataset

from tokenizer import MyTokenizer


class WikiDataset(Dataset):

    def __init__(
        self,
        file_path,
        seq_len=128
    ):

        self.tokenizer = MyTokenizer()

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()

        self.tokens = self.tokenizer.encode(
            text
        )

        self.seq_len = seq_len

    def __len__(self):

        return (
            len(self.tokens)
            // self.seq_len
        )

    def __getitem__(
        self,
        idx
    ):

        start = idx * self.seq_len

        end = start + self.seq_len

        chunk = self.tokens[
            start:end
        ]

        return torch.tensor(
            chunk,
            dtype=torch.long
        )