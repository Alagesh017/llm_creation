import numpy as np

MAX_SEQ_LEN = 512
EMBED_DIM = 384

position_embeddings = np.random.normal(
    0,
    0.02,
    (MAX_SEQ_LEN, EMBED_DIM)
).astype(np.float32)

np.save(
    "position_embeddings.npy",
    position_embeddings
)

print(position_embeddings.shape)