from datasets import load_dataset
import pandas as pd

# Stream Wikipedia instead of downloading everything
dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True
)

rows = []

# Collect first 10,000 articles
for i, article in enumerate(dataset):
    rows.append({
        "title": article["title"],
        "text": article["text"]
    })

    if i >= 9999:
        break

    if i % 1000 == 0:
        print(f"Processed {i} articles...")

# Convert to DataFrame
df = pd.DataFrame(rows)

# Save to CSV
df.to_csv("wikipedia_10000_articles.csv", index=False)

print(f"Saved {len(df)} articles to wikipedia_10000_articles.csv")