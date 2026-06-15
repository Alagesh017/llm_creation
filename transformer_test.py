import torch

from model.transformer import Transformer

x = torch.randn(
    1,
    4,
    384
)

model = Transformer()

output = model(x)

print(output.shape)