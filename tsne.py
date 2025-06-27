import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Load your fine-tuned BERT model and tokenizer
model_name = "GLiNER/logs/final_model" #"deberta_mlm_finetuned"#"bert-base-uncased" #or a path to local model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Sample texts (replace with your own data)
texts = [
    "I love this product!",
    "Terrible service, very disappointed.",
    "Amazing experience, highly recommend.",
    "Not worth the money.",
    "Satisfied with the quality.",
    "Will never buy again."
]
texts = [
    "AAAAAAAAAAAAAUUUUUUUUUUUUUG",
    "AAAAAAAAACCCCGACGUACGUAGCUAGACUAAGCUCGCUCGCUCGCUCG",
    "AUCGAUCGAUCGAUCGAUCGAUCG",
    "GCCCCCCGCCUUUUUAAAAA",
    "AAAAAAAAAAAAAUUUUUUUUUUUUUG",
    "AAAAAAAAACCCCGACGUACGUAGCUAGACUAAGCUCGCUCGCUCGCUCG",
    "AUCGAUCGAUCGAUCGAUCGAUCG",
    "GCCCCCCGCCUUUUUAAAAA"
]
def get_mean_pooled_embeddings(texts, model, tokenizer):
    embeddings = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = output.last_hidden_state  # shape: [1, seq_len, hidden_size]

            # Apply attention mask for mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # avoid division by zero
            mean_pooled = sum_embeddings / sum_mask

            embeddings.append(mean_pooled.squeeze().numpy())

    return np.array(embeddings)

# Get mean-pooled embeddings
X = get_mean_pooled_embeddings(texts, model, tokenizer)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot
plt.figure(figsize=(8, 6))
for i, label in enumerate(texts):
    plt.scatter(X_tsne[i, 0], X_tsne[i, 1])
    plt.annotate(label[:30], (X_tsne[i, 0] + 0.5, X_tsne[i, 1] + 0.5))
plt.title("t-SNE of BERT Mean-Pooled Embeddings")
plt.grid(True)
plt.show()
plt.savefig('tsne.png')
