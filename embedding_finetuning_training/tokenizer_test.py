import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from tokenizer import MyTokenizer
from model.full_model import FullModel
import torch.nn.functional as F

# ==========================================
# DEVICE
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ==========================================
# EMBEDDING MODEL CLASS
# ==========================================

class EmbeddingModel(torch.nn.Module):

    def __init__(self, vocab_size=10000):
        super().__init__()
        self.base_model = FullModel(vocab_size=vocab_size)

    def forward(self, input_ids):

        x = self.base_model.embeddings(input_ids)
        x = self.base_model.transformer(x)

        # MASKED MEAN POOLING
        mask = (input_ids != 0).unsqueeze(-1).float()

        x = x * mask
        x = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # NORMALIZE
        x = F.normalize(x, p=2, dim=1)

        return x

# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading model...")

model = EmbeddingModel(vocab_size=10000)

state_dict_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "my_model.pt"
)

state_dict = torch.load(
    state_dict_path,
    map_location=DEVICE,
    weights_only=True
)

model.base_model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully!")

# ==========================================
# TOKENIZER
# ==========================================

tokenizer = MyTokenizer()

# ==========================================
# TEST INPUT
# ==========================================

text = "A man is playing guitar"

print("\nInput Text:", text)

ids = tokenizer.encode(text)
ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

# ==========================================
# EMBEDDING INFERENCE
# ==========================================

with torch.no_grad():
    emb = model(ids)

# ==========================================
# OUTPUT
# ==========================================

print("\nEmbedding Shape:", emb.shape)
print("First 10 values:", emb[0][:10])