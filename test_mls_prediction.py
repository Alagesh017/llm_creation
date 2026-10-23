import os

import torch

from tokenizer import MyTokenizer
from model.full_model import FullModel

# ==========================================
# Configuration
# ==========================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "my_model.pt"
)

VOCAB_SIZE = 10000

MASK_TOKEN_ID = 4

TOP_K = 10

# ==========================================
# Sentence + Word To Predict
# ==========================================

sentence = "he is a boy"

word_to_mask = "is"

# ==========================================
# Load Tokenizer
# ==========================================

tokenizer = MyTokenizer()

# ==========================================
# Load Model
# ==========================================

model = FullModel(
    vocab_size=VOCAB_SIZE
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

print("Model Loaded Successfully\n")

# ==========================================
# Show Original Tokens
# ==========================================

print("Original Sentence:")
print(sentence)

print("\nOriginal Tokens:\n")

original_ids = tokenizer.encode(sentence)

for i, token_id in enumerate(original_ids):
    print(f"{i} -> {tokenizer.decode([token_id])!r}")

# ==========================================
# Mask The Chosen Word
# ==========================================

masked_sentence = sentence.replace(word_to_mask, "[MASK]")

ids = tokenizer.encode(masked_sentence)

if MASK_TOKEN_ID not in ids:
    raise ValueError(
        "Could not find [MASK] token. "
        "Make sure the tokenizer recognizes '[MASK]' as a special token."
    )

mask_pos = ids.index(MASK_TOKEN_ID)

print("\nMasked Sentence:")
print(masked_sentence)

print("\nMasked Tokens:\n")

for i, token_id in enumerate(ids):
    print(f"{i} -> {tokenizer.decode([token_id])!r}")

# ==========================================
# Forward Pass
# ==========================================

input_ids = torch.tensor([ids])

with torch.no_grad():
    logits = model(input_ids)

# ==========================================
# Top-K Predictions
# ==========================================

topk = torch.topk(logits[0, mask_pos], k=TOP_K)

print(f"\nTop {TOP_K} Predictions For Position {mask_pos}:\n")

for rank, idx in enumerate(topk.indices):
    word = tokenizer.decode([idx.item()])
    score = topk.values[rank].item()
    print(f"{rank+1}. {word!r} (score={score:.4f})")

# ==========================================
# Reference
# ==========================================

print(f"\nOriginal word was: {word_to_mask!r}")
