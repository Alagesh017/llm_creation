import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import torch

from tokenizer import MyTokenizer
from model.full_model import FullModel

# ==========================================
# Load Tokenizer
# ==========================================

tokenizer = MyTokenizer()

# ==========================================
# Load Model
# ==========================================

model = FullModel(
    vocab_size=10000
)

model.load_state_dict(
    torch.load(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "my_model.pt"
        ),
        map_location="cpu"
    )
)

model.eval()

# ==========================================
# Input Text
# ==========================================

text = "Earth revolves around the Sun"

ids = tokenizer.encode(text)

print("\nOriginal Tokens:\n")

for i, token_id in enumerate(ids):
    print(
        f"{i} -> {tokenizer.decode([token_id])}"
    )

# ==========================================
# Mask "programming"
# ==========================================

mask_pos = 8
ids[mask_pos] = 4

print("\nMasked Sentence Tokens:\n")

for i, token_id in enumerate(ids):
    print(
        f"{i} -> {tokenizer.decode([token_id])}"
    )

# ==========================================
# Forward Pass
# ==========================================

input_ids = torch.tensor([ids])

with torch.no_grad():

    logits = model(input_ids)

# ==========================================
# Top Predictions
# ==========================================

topk = torch.topk(
    logits[0, mask_pos],
    k=10
)

print("\nTop 10 Predictions:\n")

for rank, idx in enumerate(topk.indices):

    word = tokenizer.decode(
        [idx.item()]
    )

    score = topk.values[
        rank
    ].item()

    print(
        f"{rank+1}. {word} "
        f"(score={score:.4f})"
    )