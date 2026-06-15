import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from tokenizer import MyTokenizer

# ==========================
# Load Tokenizer
# ==========================

tokenizer = MyTokenizer()

# ==========================
# Test Sentence
# ==========================

text = "Earth revolves around the Sun"

# Encode
ids = tokenizer.encode(text)

# ==========================
# Show Tokens
# ==========================

print("\nSentence:")
print(text)

print("\nToken IDs:")
print(ids)

print("\nDecoded Tokens:\n")

for i, token_id in enumerate(ids):

    token = tokenizer.decode(
        [token_id]
    )

    print(
        f"{i} -> {repr(token)}"
    )