import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.full_model import FullModel

# ==========================================
# CONFIG
# ==========================================

VOCAB_SIZE = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

# ==========================================
# EMBEDDING MODEL
# ==========================================

class EmbeddingModel(nn.Module):

    def __init__(self, vocab_size=10000):
        super().__init__()

        self.base_model = FullModel(vocab_size=vocab_size)

    def forward(self, input_ids):

        x = self.base_model.embeddings(input_ids)
        x = self.base_model.transformer(x)

        # MASKED MEAN POOLING (IMPORTANT)
        mask = (input_ids != 0).unsqueeze(-1).float()

        x = x * mask
        x = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        x = F.normalize(x, p=2, dim=1)

        return x

# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading pretrained MLM model...")

model = EmbeddingModel(vocab_size=VOCAB_SIZE)

state_dict = torch.load(os.path.join(os.path.dirname(__file__), "..", "my_model.pt"), map_location=DEVICE,weights_only=True)

model.base_model.load_state_dict(state_dict)

model = model.to(DEVICE)

print("Model loaded successfully!")

# ==========================================
# TEST FUNCTION (VERY IMPORTANT)
# ==========================================

def get_embedding(text, tokenizer):

    model.eval()

    tokens = tokenizer.encode(text)

    input_ids = torch.tensor(tokens, dtype=torch.long)
    input_ids = input_ids.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(input_ids)

    return emb.cpu().numpy()

# ==========================================
# SAVE EMBEDDING MODEL
# ==========================================

def save_embedding_model(path="embedding_model.pt"):

    torch.save(model.state_dict(), path)
    print(f"Saved embedding model → {path}")

# ==========================================
# OPTIONAL RUN CHECK
# ==========================================

if __name__ == "__main__":

    print("\nModel ready for embedding inference!")

    # Example test (replace tokenizer with yours)
    # from tokenizer import MyTokenizer
    # tokenizer = MyTokenizer()
    # print(get_embedding("hello world", tokenizer).shape)