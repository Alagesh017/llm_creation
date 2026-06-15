
from model.mlm_dataset import MLMDataset


dataset = MLMDataset(
    "data/training_text.txt"
)

input_ids, labels = dataset[0]

print(input_ids.shape)
print(labels.shape)

print(input_ids[:20])
print(labels[:20])