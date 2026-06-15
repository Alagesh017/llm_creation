import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from tokenizer import MyTokenizer
from model.full_model import FullModel

# ==========================================
# CONFIG
# ==========================================

BATCH_SIZE = 16          # Bigger batch = more stable gradient signal
EPOCHS = 5
LEARNING_RATE = 2e-5     # Much higher — model was barely moving at 5e-7
VOCAB_SIZE = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nDevice: {device}")

# ==========================================
# DATASET
# ==========================================

class STSDataset(Dataset):

    def __init__(self, split="train"):
        self.dataset = load_dataset("mteb/stsbenchmark-sts")[split]
        self.tokenizer = MyTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]

        s1 = item["sentence1"]
        s2 = item["sentence2"]

        score = item["score"] / 5.0  # Normalise to [0, 1]

        ids1 = self.tokenizer.encode(s1)
        ids2 = self.tokenizer.encode(s2)

        return (
            torch.tensor(ids1, dtype=torch.long),
            torch.tensor(ids2, dtype=torch.long),
            torch.tensor(score, dtype=torch.float)
        )

# ==========================================
# COLLATE
# ==========================================

def collate_fn(batch):

    ids1, ids2, scores = zip(*batch)

    max_len1 = max(len(x) for x in ids1)
    max_len2 = max(len(x) for x in ids2)

    padded1, padded2 = [], []

    for x in ids1:
        pad_len = max_len1 - len(x)
        padded1.append(F.pad(x, (0, pad_len), value=0))

    for x in ids2:
        pad_len = max_len2 - len(x)
        padded2.append(F.pad(x, (0, pad_len), value=0))

    return (
        torch.stack(padded1),
        torch.stack(padded2),
        torch.stack(scores)
    )

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

        # Masked mean pooling (ignore padding tokens)
        mask = (input_ids != 0).unsqueeze(-1).float()
        x = x * mask
        x = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        x = F.normalize(x, p=2, dim=1)

        return x

# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading Model...")

model = EmbeddingModel(vocab_size=VOCAB_SIZE)

state_dict = torch.load(
    "my_model.pt",
    map_location=device,
    weights_only=True
)

model.base_model.load_state_dict(state_dict)
model = model.to(device)

print("Model Loaded ✔")

# ==========================================
# DATA LOADERS
# ==========================================

train_dataset = STSDataset(split="train")
dev_dataset   = STSDataset(split="validation")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# ==========================================
# OPTIMIZER + SCHEDULER
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

# Cosine LR decay across all steps
total_steps = EPOCHS * len(train_loader)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=1e-7
)

# ==========================================
# LOSS FUNCTION
#
# WHY we replaced contrastive loss:
#
# 1. STS has GRADED scores (0.0 → 1.0), not hard positive/negative pairs.
#    Contrastive/InfoNCE losses assume each sample has exactly one positive —
#    that's wrong here. A pair with score=0.8 is "more positive" than score=0.3.
#
# 2. The old contrastive_loss had broken labels:
#    labels = [0,1,2,..., 0,1,2,...] but the sim matrix is 2B×2B with emb1 and
#    emb2 concatenated — emb1[i] should match emb2[i] (at index i+B), not itself.
#    This caused random cross_entropy which exploded with temperature=0.1.
#
# FIX: Pure MSE on cosine similarity vs. target score.
#      Simple, directly optimises what we care about, stable gradients.
# ==========================================

mse_loss = nn.MSELoss()

# ==========================================
# EVAL FUNCTION (Pearson correlation on dev set)
# ==========================================

def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for ids1, ids2, targets in loader:
            ids1 = ids1.to(device)
            ids2 = ids2.to(device)

            emb1 = model(ids1)
            emb2 = model(ids2)

            preds = F.cosine_similarity(emb1, emb2)

            all_preds.append(preds.cpu())
            all_targets.append(targets)

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Pearson correlation
    vx = all_preds   - all_preds.mean()
    vy = all_targets - all_targets.mean()
    pearson = (vx * vy).sum() / (
        (vx ** 2).sum().sqrt() * (vy ** 2).sum().sqrt() + 1e-9
    )

    mse = F.mse_loss(all_preds, all_targets).item()

    return pearson.item(), mse

# ==========================================
# TRAINING
# ==========================================

print("\nTraining Started...\n")

best_pearson  = -1.0
best_epoch    = 0

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for step, (ids1, ids2, target_scores) in enumerate(train_loader):

        ids1          = ids1.to(device)
        ids2          = ids2.to(device)
        target_scores = target_scores.to(device).float()

        optimizer.zero_grad()

        emb1 = model(ids1)
        emb2 = model(ids2)

        cosine_sim = F.cosine_similarity(emb1, emb2)

        # Direct MSE between predicted cosine sim and target score
        loss = mse_loss(cosine_sim, target_scores)

        loss.backward()

        # Gradient clipping (keeps training stable early on)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if step % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch+1}/{EPOCHS}  "
                f"Step {step:>4}/{len(train_loader)}  "
                f"Loss {loss.item():.4f}  "
                f"LR {lr_now:.2e}"
            )

    avg_train_loss = total_loss / len(train_loader)

    # Evaluate on dev set
    pearson, dev_mse = evaluate(model, dev_loader)
    print(
        f"\n── Epoch {epoch+1} Summary ──────────────────────\n"
        f"   Train avg loss : {avg_train_loss:.4f}\n"
        f"   Dev MSE        : {dev_mse:.4f}\n"
        f"   Dev Pearson r  : {pearson:.4f}\n"
        f"────────────────────────────────────────────────\n"
    )

    # Save best checkpoint
    if pearson > best_pearson:
        best_pearson = pearson
        best_epoch   = epoch + 1
        torch.save(model.state_dict(), "embedding_model_best.pt")
        print(f"   ✔ Best model saved (epoch {best_epoch}, r={best_pearson:.4f})\n")

# ==========================================
# FINAL SAVE
# ==========================================

torch.save(model.state_dict(), "embedding_model.pt")

print(f"\nDone. Best Pearson r={best_pearson:.4f} at epoch {best_epoch}")
print("Saved: embedding_model.pt  +  embedding_model_best.pt")