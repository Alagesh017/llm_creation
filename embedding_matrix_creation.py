import json
import numpy as np

# Load vocab
with open("my_tokenizer/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

VOCAB_SIZE = len(vocab)
EMBED_DIM = 384

# Random initialization
embedding_matrix = np.random.normal(
    loc=0.0,
    scale=0.02,
    size=(VOCAB_SIZE, EMBED_DIM)
).astype(np.float32)

# Save
np.save("embedding_matrix.npy", embedding_matrix)

print("Vocabulary Size:", VOCAB_SIZE)
print("Embedding Shape:", embedding_matrix.shape)