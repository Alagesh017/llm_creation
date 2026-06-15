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
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import MyTokenizer
from model.full_model import FullModel

# ==========================================
# Embedding Model Class
# ==========================================

class EmbeddingModel(nn.Module):

    def __init__(self, vocab_size=10000):
        super().__init__()
        self.base_model = FullModel(vocab_size=vocab_size)

    def forward(self, input_ids):

        x = self.base_model.embeddings(input_ids)
        x = self.base_model.transformer(x)

        # Masked Mean Pooling
        mask = (input_ids != 0).unsqueeze(-1).float()

        x = x * mask
        x = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        x = F.normalize(x, p=2, dim=1)

        return x

# ==========================================
# Config
# ==========================================

VOCAB_SIZE = 10000
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "embedding_model.pt")

# ==========================================
# Load Tokenizer
# ==========================================

tokenizer = MyTokenizer()

# ==========================================
# Load Model
# ==========================================

model = EmbeddingModel(vocab_size=VOCAB_SIZE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

print("Model Loaded Successfully\n")

# ==========================================
# Embedding Function
# ==========================================

def get_embedding(text):

    ids = tokenizer.encode(text)

    input_ids = torch.tensor(
        [ids],
        dtype=torch.long
    )

    with torch.no_grad():
        embedding = model(input_ids)

    return embedding

# ==========================================
# Test Sentences
# ==========================================

text1 = "Python is a programming language"

text2 = "Python is used for software development"

text3 = "Paris is the capital of France"

print("Sentence 1:")
print(text1)

print("\nSentence 2:")
print(text2)

print("\nSentence 3:")
print(text3)

# ==========================================
# Generate Embeddings
# ==========================================

emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)

# ==========================================
# Cosine Similarity
# ==========================================

sim_12 = F.cosine_similarity(
    emb1,
    emb2
).item()

sim_13 = F.cosine_similarity(
    emb1,
    emb3
).item()

sim_23 = F.cosine_similarity(
    emb2,
    emb3
).item()

# ==========================================
# Results
# ==========================================

print("\n============================")
print("COSINE SIMILARITY RESULTS")
print("============================\n")

print(
    f"Python vs Software : {sim_12:.4f}"
)

print(
    f"Python vs Paris    : {sim_13:.4f}"
)

print(
    f"Software vs Paris  : {sim_23:.4f}"
)

print("\n============================")

if sim_12 > sim_13:
    print(
        "\nGOOD: Related sentences are closer."
    )
else:
    print(
        "\nWARNING: Embeddings need more training."
    )