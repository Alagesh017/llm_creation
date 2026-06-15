from dataset import WikiDataset

dataset = WikiDataset(
    "data/training_text.txt",
    seq_len=128
)

print(
    "Samples:",
    len(dataset)
)

print(
    dataset[0].shape
)