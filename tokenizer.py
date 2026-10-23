from tokenizers import ByteLevelBPETokenizer
import os


class MyTokenizer:

    def __init__(self):
        # Resolve paths relative to the project root
        project_root = os.path.abspath(os.path.dirname(__file__))
        vocab_path = os.path.join(project_root, "my_tokenizer/vocab.json")
        merges_path = os.path.join(project_root, "my_tokenizer/merges.txt")

        self.tokenizer = ByteLevelBPETokenizer(
            vocab_path,
            merges_path
        )

        # Re-register special tokens. They are not restored automatically
        # when loading from vocab.json / merges.txt, so "[MASK]" etc. would
        # otherwise be split into raw bytes instead of their single id.
        self.tokenizer.add_special_tokens([
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]"
        ])

    def encode(self, text):

        return self.tokenizer.encode(text).ids

    def decode(self, ids):

        return self.tokenizer.decode(ids)