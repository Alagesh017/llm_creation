import sys
import os

# ==========================================
# Add Project Root to Python Path
# ==========================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

# ==========================================
# Imports
# ==========================================

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from model.full_model import FullModel
from model.mlm_dataset import MLMDataset

# ==========================================
# Configuration
# ==========================================

BATCH_SIZE = 24
EPOCHS = 1
MAX_STEPS_PER_EPOCH = 200
LEARNING_RATE = 1e-4
VOCAB_SIZE = 10000

# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

# ==========================================
# Dataset
# ==========================================

print("\nLoading Dataset...")

dataset = MLMDataset(
    "data/training_text.txt"
)

print(
    f"Total Samples: {len(dataset)}"
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ==========================================
# Model
# ==========================================

print("\nLoading Model...")

model = FullModel(
    vocab_size=VOCAB_SIZE
).to(device)

# ==========================================
# Loss Function
# ==========================================

criterion = nn.CrossEntropyLoss(
    ignore_index=-100
)

# ==========================================
# Optimizer
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================================
# Training Loop
# ==========================================

print("\nTraining Started...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for step, (input_ids, labels) in enumerate(
        dataloader
    ):

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids
        )

        loss = criterion(
            outputs.view(
                -1,
                VOCAB_SIZE
            ),
            labels.view(-1)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        # ----------------------------------
        # Print Progress Every 100 Steps
        # ----------------------------------

        if step % 100 == 0:

            print(
                f"Epoch "
                f"{epoch+1}/{EPOCHS} | "
                f"Step "
                f"{step}/{len(dataloader)} | "
                f"Loss: "
                f"{loss.item():.4f}"
            )

        # Limit steps per epoch
        if step + 1 >= MAX_STEPS_PER_EPOCH:
            break

    avg_loss = (
        total_loss /
        min(len(dataloader), MAX_STEPS_PER_EPOCH)
    )

    print(
        f"\nEpoch "
        f"{epoch+1} Completed"
    )

    print(
        f"Average Loss: "
        f"{avg_loss:.4f}\n"
    )

    # =====================================
    # Save Checkpoint
    # =====================================

    checkpoint_name = (
        f"checkpoint_epoch_"
        f"{epoch+1}.pt"
    )

    torch.save(
        model.state_dict(),
        checkpoint_name
    )

    print(
        f"Saved: "
        f"{checkpoint_name}\n"
    )

# ==========================================
# Final Save
# ==========================================

torch.save(
    model.state_dict(),
    "my_model.pt"
)

print(
    "\nTraining Finished!"
)

print(
    "Model Saved As: my_model.pt"
)