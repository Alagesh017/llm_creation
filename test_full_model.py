import torch

from model.full_model import FullModel




model = FullModel()

input_ids = torch.randint(
    0,
    10000,
    (1, 128)
)

output = model(
    input_ids
)

print(
    output.shape
)