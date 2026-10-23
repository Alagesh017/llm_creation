from itertools import islice

import pandas as pd
from datasets import load_dataset

NUM_ARTICLES = 3000
OUTPUT_FILE = f"wikipedia_{NUM_ARTICLES}_articles.csv"

# Stream Wikipedia instead of downloading everything
dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True,
)

rows = []

# Collect only the first NUM_ARTICLES articles
for i, article in enumerate(islice(dataset, NUM_ARTICLES), start=1):
    rows.append({
        "title": article["title"],
        "text": article["text"],
    })

    if i % 500 == 0:
        print(f"Processed {i} articles...")

# Convert to DataFrame and save to CSV
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(df)} articles to {OUTPUT_FILE}")