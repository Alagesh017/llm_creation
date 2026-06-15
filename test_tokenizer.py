from tokenizer import MyTokenizer

tokenizer = MyTokenizer()

text = "Hello world"

ids = tokenizer.encode(text)

print("Token IDs:")
print(ids)

decoded = tokenizer.decode(ids)

print("\nDecoded:")
print(decoded)