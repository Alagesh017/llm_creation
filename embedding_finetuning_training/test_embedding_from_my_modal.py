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
print("\nDevice:", DEVICE)

# ==========================================
# EMBEDDING MODEL WRAPPER
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

        x = F.normalize(x, p=2, dim=1)

        return x

# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading model...")

model = EmbeddingModel(vocab_size=10000)

model_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "my_model.pt"
)

state_dict = torch.load(
    model_path,
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
# TEST SENTENCES
# ==========================================

sentences = [
    "A man is playing guitar",
    "A person is playing music",
    "A dog is running in the park",
    "Quantum physics is very complex",
    "The cat is sleeping on the sofa"
]

# ==========================================
# EMBEDDING FUNCTION
# ==========================================

def get_embedding(text):
    ids = tokenizer.encode(text)
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(ids)

    return emb

# ==========================================
# RUN TESTS
# ==========================================

print("\n=============================")
print("EMBEDDING TEST START")
print("=============================\n")

embeddings = []

for text in sentences:

    emb = get_embedding(text)
    embeddings.append(emb)

    print("Text:", text)
    print("Shape:", emb.shape)
    print("Mean:", emb.mean().item())
    print("Std:", emb.std().item())
    print("First 5 values:", emb[0][:5])
    print("-----------------------------")

# ==========================================
# SIMILARITY TEST
# ==========================================

print("\n=============================")
print("SIMILARITY TEST")
print("=============================\n")

def cosine(a, b):
    return F.cosine_similarity(a, b).item()

pairs = [
    (0, 1),  # similar
    (0, 2),  # different
    (2, 3),  # very different
    (0, 4)   # unrelated
]

for i, j in pairs:

    sim = cosine(embeddings[i], embeddings[j])

    print(f"Similarity between:")
    print(f"  - {sentences[i]}")
    print(f"  - {sentences[j]}")
    print("Score:", sim)
    print("-----------------------------")

# ==========================================
# FINAL CHECK
# ==========================================

print("\n=============================")
print("FINAL MODEL HEALTH CHECK")
print("=============================\n")

print("Embedding dimension:", embeddings[0].shape)
print("All tests completed!")