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
import torch.nn.functional as F
import numpy as np

from tokenizer import MyTokenizer
from model.full_model import FullModel

# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "my_model.pt"
VOCAB_SIZE = 10000

# ==========================================
# Load Tokenizer
# ==========================================

print("Loading Tokenizer...")

tokenizer = MyTokenizer()

# ==========================================
# Load Model
# ==========================================

print("Loading Model...")

model = FullModel(
    vocab_size=VOCAB_SIZE
)

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

        # ------------------------------
        # Embedding Layer
        # ------------------------------

        x = model.embeddings(
            input_ids
        )

        # ------------------------------
        # Transformer Encoder
        # ------------------------------

        x = model.transformer(
            x
        )

        # ------------------------------
        # Mean Pooling
        # ------------------------------

        embedding = x.mean(
            dim=1
        )

        # ------------------------------
        # L2 Normalize
        # ------------------------------

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

    return embedding.squeeze().cpu().numpy()

# ==========================================
# Input Paragraph
# ==========================================

paragraph = input(
    "Enter Paragraph:\n\n"
)

# ==========================================
# Generate Embedding
# ==========================================

vector = get_embedding(
    paragraph
)

# ==========================================
# Results
# ==========================================

print("\nEmbedding Shape:")
print(vector.shape)

print("\nFirst 20 Values:\n")

print(vector[:20])

# ==========================================
# Save Vector
# ==========================================

np.save(
    "embedding_vector.npy",
    vector
)

print(
    "\nEmbedding Saved As:"
)

print(
    "embedding_vector.npy"
)