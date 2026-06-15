from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer(
    "my_tokenizer/vocab.json",
    "my_tokenizer/merges.txt"
)

print(tokenizer.token_to_id("[MASK]"))