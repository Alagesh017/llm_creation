# Masked LLM From Scratch

A BERT-style masked language model built from scratch (pure PyTorch) and then fine-tuned into a sentence-embedding model.

Pipeline: **Wikipedia data → BPE tokenizer → embedding tables → MLM pre-training → STS fine-tuning → similarity inference**.

- Architecture: Transformer encoder, 6 layers, `embed_dim=384`, `num_heads=12`, `hidden_dim=1536`
- Tokenizer: Byte-Level BPE, `vocab_size=10000`
- Vocab size used by the model: `10000`
- Embedding dimension: `384`
- Max sequence length: `512` (`seq_len=128` used for MLM chunks)

---

## 1. Requirements

Python 3.10+ and the following packages:

```bash
pip install torch numpy pandas datasets tokenizers
```

Check that your GPU is visible (optional, training falls back to CPU):

```bash
python verify_gpu.py
```

> Run **every** command below from the project root (`d:\llm_creation`). Most scripts read/write paths relative to the current working directory.

---

## 2. Project Structure

```
llm_creation/
├── data_set_retrival.py              # Step 1 – download Wikipedia data
├── tokenizer_train.py                # Step 2 – build text file + train BPE tokenizer
├── tokenizer.py                      # MyTokenizer wrapper (used everywhere)
├── embedding_matrix_creation.py      # Step 3a – token embedding table
├── positioning_matrix.py             # Step 3b – position embedding table
├── test_mls_prediction.py            # Step 7 – mask a word and predict it (run last)
│
├── model/                            # The Transformer, built from scratch
│   ├── embeddings.py                 # token + position embeddings
│   ├── attention.py                  # multi-head self attention
│   ├── encoder.py                    # encoder block (attention + FFN + layernorm)
│   ├── feedforward.py                # position-wise FFN
│   ├── layernorm.py
│   ├── transformer.py                # stack of 6 encoder blocks
│   ├── mlm_head.py                   # MLM prediction head
│   ├── full_model.py                 # embeddings + transformer + mlm_head
│   └── mlm_dataset.py                # dataset that does 15% random masking
│
├── train/
│   └── train.py                      # Step 4 – MLM pre-training
│
├── embedding_finetuning_training/
│   ├── train_embedding.py            # Step 5 – STS fine-tuning (MSE on cosine sim)
│   ├── test_embedding_from_my_modal.py
│   ├── modal_cases_test.py
│   └── tokenizer_test.py
│
├── test_mlm/                         # Step 6 – inference / sanity checks
│   ├── test_mlm.py                   # predict masked token
│   ├── get_masked.py                 # show tokens for a sentence
│   ├── generate_embedding.py         # text -> embedding vector
│   └── text_similarities.py          # cosine similarity between sentences
│
├── Training_sentence_similarities/
│   └── dataset_test.py               # peek at the STS benchmark dataset
│
├── data/
│   └── training_text.txt             # training corpus (plain text)
├── my_tokenizer/
│   ├── vocab.json
│   └── merges.txt
│
├── embedding_matrix.npy              # generated
├── position_embeddings.npy           # generated
├── my_model.pt                       # generated – pre-trained MLM
├── embedding_model.pt                # generated – fine-tuned embeddings
└── embedding_model_best.pt           # generated – best fine-tuned checkpoint
```

---

## 3. Full Run – Step by Step

### Step 1 — Download the data

Streams the first `NUM_ARTICLES` English Wikipedia articles (set to `3000` in the script) and saves them to CSV.

```bash
python data_set_retrival.py
```

Output: `wikipedia_3000_articles.csv` (rows with `title`, `text`).

### Step 2 — Build the corpus text + train the tokenizer

Creates `training_text.txt` from the CSV text column, then trains a Byte-Level BPE tokenizer with the special tokens `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`.

```bash
python tokenizer_train.py
```

Output: `training_text.txt`, `my_tokenizer/vocab.json`, `my_tokenizer/merges.txt`.

> The pre-training dataset reads its corpus from `data/training_text.txt`, so move/copy the generated file into `data/`:
>
> ```bash
> move training_text.txt data\training_text.txt
> ```

Quick sanity check (prints the token id of `[MASK]`, expected `4`):

```bash
python actual_mask_token.py
python verify_token.py
```

### Step 3 — Create the embedding tables

Both are randomly initialised and become trainable parameters during pre-training.

```bash
python embedding_matrix_creation.py   # embedding_matrix.npy  -> (10000, 384)
python positioning_matrix.py          # position_embeddings.npy -> (512, 384)
```

### Step 4 — MLM pre-training

Loads `data/training_text.txt`, masks 15% of tokens at random, and trains `FullModel` with CrossEntropy (`ignore_index=-100`). `BATCH_SIZE`, `EPOCHS`, `MAX_STEPS_PER_EPOCH`, and `LEARNING_RATE` are configured at the top of the script.

```bash
python train/train.py
```

Output: `checkpoint_epoch_*.pt` after each epoch, and `my_model.pt` at the end.

### Step 5 — Fine-tune into a sentence-embedding model

Loads `my_model.pt` as the base, adds masked-mean pooling + L2 normalisation, and trains on `mteb/stsbenchmark-sts` by minimising MSE between predicted cosine similarity and the human gold score. 5 epochs, batch size 16, lr `2e-5`, cosine LR decay, gradient clipping.

```bash
python embedding_finetuning_training/train_embedding.py
```

Output: `embedding_model.pt` and `embedding_model_best.pt` (highest dev Pearson r).

### Step 6 — Inference & sanity checks

**Inspect how a sentence is tokenized:**

```bash
python test_mlm/get_masked.py
```

**Generate an embedding vector for a paragraph** (prompts for input, saves `embedding_vector.npy`):

```bash
python test_mlm/generate_embedding.py
```

**Compare sentence similarity** (uses the fine-tuned `embedding_model.pt`):

```bash
python test_mlm/text_similarities.py
```

**Embedding health check** (similar vs. unrelated sentence scores):

```bash
python embedding_finetuning_training/test_embedding_from_my_modal.py
python embedding_finetuning_training/modal_cases_test.py
```

**Peek at the STS benchmark data:**

```bash
python Training_sentence_similarities/dataset_test.py
```

---

### Step 7 — Predict a masked word (run this last)

Edit the `sentence` and `word_to_mask` at the top of `test_mls_prediction.py`, then run:

```bash
python test_mls_prediction.py
```

It swaps the chosen word for `[MASK]`, runs the pre-trained `my_model.pt`, and prints the **top 10 predictions** for that slot. Prediction quality depends on how long Step 4 was trained.

---

## 4. Quick Reference – Order of Operations

| # | Command | Produces |
|---|---------|----------|
| 1 | `python data_set_retrival.py` | `wikipedia_3000_articles.csv` |
| 2 | `python tokenizer_train.py` | `training_text.txt`, `my_tokenizer/*` |
| 2b | `move training_text.txt data\training_text.txt` | corpus for training |
| 3 | `python embedding_matrix_creation.py` | `embedding_matrix.npy` |
| 3b | `python positioning_matrix.py` | `position_embeddings.npy` |
| 4 | `python train/train.py` | `my_model.pt` |
| 5 | `python embedding_finetuning_training/train_embedding.py` | `embedding_model_best.pt` |
| 6 | `python test_mlm/text_similarities.py` | similarity scores |
| 7 | `python test_mls_prediction.py` | top-10 masked-word predictions |

---

## 5. Other Utility Scripts

| Script | Purpose |
|--------|---------|
| `verify_gpu.py` | Confirms CUDA is available and prints the GPU name |
| `test_full_model.py` | Forward pass shape check on random input |
| `transformer_test.py` | Forward pass shape check for the encoder stack |
| `test_dataset.py` | Loads `WikiDataset` and prints sample count / shape |
| `test_dataset.py` / `mask_dataset_test.py` | Verifies the MLM masking produces `input_ids` + `labels` |
| `trial.py` | Standalone pure-Python sentiment classifier (no torch/numpy) used to learn the math |

---

## 6. Notes & Troubleshooting

- **Run from the project root.** `my_tokenizer/`, `embedding_matrix.npy`, `position_embeddings.npy`, and `my_model.pt` are resolved relative to the repo root.
- **The corpus path matters.** Pre-training reads `data/training_text.txt` — make sure Step 2's output ends up there.
- **No GPU?** Everything falls back to CPU automatically, but pre-training will be slow. `BATCH_SIZE`, `EPOCHS`, and `LEARNING_RATE` are defined at the top of `train/train.py` and `embedding_finetuning_training/train_embedding.py` if you want to tune them.
- **`[MASK]` token id is `4`** because the special tokens are added in order `[PAD], [UNK], [CLS], [SEP], [MASK]`.
- **Order matters:** you must complete Steps 1–4 before Step 5, since fine-tuning loads `my_model.pt`.
