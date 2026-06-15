from datasets import load_dataset

ds = load_dataset("mteb/stsbenchmark-sts")
from datasets import load_dataset

ds = load_dataset("mteb/stsbenchmark-sts")

for i in range(10):
    print("\n------------------")
    print(ds["train"][i]["sentence1"])
    print(ds["train"][i]["sentence2"])
    print("Score:", ds["train"][i]["score"])