from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
import re
import json

# Load vocab
with open("rnafm_vocab.json", "r") as f:
    vocab = json.load(f)

# Create tokenizer
tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))

# Use regex to split every character (1-mer)
tokenizer.pre_tokenizer = Split(pattern=r"", behavior="isolated")

# Save tokenizer (this works)
tokenizer.save("tokenizer.json")

from transformers import PreTrainedTokenizerFast

# Load tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer.json",  # From tokenizers lib
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<cls>",
    eos_token="<eos>",
    mask_token="<mask>",
    additional_special_tokens=["<null>"],
    sep_token="<eos>"  # Optional, same as eos
)

# Set model max length
tokenizer.model_max_length = 1024
tokenizer.save_pretrained("rnafm-fast-tokenizer")
from transformers import PreTrainedTokenizerFast

rna_tokenizer = PreTrainedTokenizerFast.from_pretrained("rnafm-fast-tokenizer")

print(rna_tokenizer.tokenize("ACCC3"))
print(rna_tokenizer.special_tokens_map)
print(rna_tokenizer.additional_special_tokens)

