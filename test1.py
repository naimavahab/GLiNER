from transformers import AutoModel
import torch

model_name = "deberta_mlm_finetuned"
model = AutoModel.from_pretrained(model_name, local_files_only=True)

# Get the embeddings layer
embeddings = model.get_input_embeddings()  # nn.Embedding

# Get vocab size and embedding dim
vocab_size, embedding_dim = embeddings.weight.shape
print(f"Model embedding vocab size: {vocab_size}")
print(f"Embedding dimension: {embedding_dim}")

# Print embedding vector shape for first 10 token IDs
for token_id in range(min(10, vocab_size)):
    embedding_vector = embeddings.weight[token_id]
    print(f"Token ID {token_id} embedding vector shape: {embedding_vector.shape}")
    # optionally print first few values of vector
    print(f"Token ID {token_id} embedding values (first 5 dims): {embedding_vector[:5].tolist()}")

