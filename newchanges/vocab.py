tokens = [
    "<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>",
    "A", "C", "G", "U", "N", "R", "Y", "S", "W", "K", "M",
    "B", "D", "H", "V", ".", "X", "*", "-", "I"
]

# Build vocab dictionary
vocab_dict = {token: idx for idx, token in enumerate(tokens)}

# Save as vocab.json
import json
with open("rnafm_vocab.json", "w") as f:
    json.dump(vocab_dict, f)


