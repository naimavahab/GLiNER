from transformers import AutoModel, AutoTokenizer

model_name = "deberta_mlm_finetuned"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Check vocab size of tokenizer and model embeddings
tokenizer_vocab_size = len(tokenizer)
model_vocab_size = model.get_input_embeddings().num_embeddings

print(f"Tokenizer vocab size: {tokenizer_vocab_size}")
print(f"Model embedding vocab size: {model_vocab_size}")

# Make sure sizes match (usually they do)
assert tokenizer_vocab_size == model_vocab_size

# Create a reverse vocab dict {id: token}
id_to_token = {idx: tok for tok, idx in tokenizer.get_vocab().items()}

# Print first 50 tokens by id order
for i in range(50):
    print(f"Token id {i}: {id_to_token[i]}")

