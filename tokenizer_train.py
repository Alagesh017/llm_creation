import os
import pandas as pd
from tokenizers import ByteLevelBPETokenizer

# Create output folder
os.makedirs("my_tokenizer", exist_ok=True)

# Load CSV
df = pd.read_csv("wikipedia_10000_articles.csv")

# Save text column
with open("training_text.txt", "w", encoding="utf-8") as f:
    for text in df["text"].dropna():
        f.write(str(text) + "\n")

print("Text file created")

# Train tokenizer
tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=["training_text.txt"],
    vocab_size=10000,
    min_frequency=2,
    special_tokens=[
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]"
    ]
)

# Save tokenizer
tokenizer.save_model("my_tokenizer")

print("Tokenizer trained successfully!")