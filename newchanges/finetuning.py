import torch
from transformers import Trainer, TrainingArguments
from multimolecule import RnaTokenizer, RnaFmForTokenPrediction
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import json

# === Label Setup ===
label2id = {
    "O": 0,
    "HAIRPIN": 1,
    "BULGE": 2,
    "INTERNAL": 3,
    "MULTILOOP": 4,
    "EXTERNAL": 5,
    "UNSTRUCTURED": 6
}

id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)

'''
# === Example Data ===
examples = [
    {
        "text": "UAGCUUAUCAGACUGAUGUUG",
        "labels": ["O", "O", "HAIRPIN", "HAIRPIN", "O", "O", "BULGE", "BULGE", "O", "O", 
                   "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
    },
    # Add more annotated examples here
]
'''
# Load data
with open("rna_substructure_examples.json", "r") as f:
    raw_examples = json.load(f)

# Keep only the labels that your model can handle
allowed_labels = set(label2id.keys())

examples = []
for entry in raw_examples:
    if len(entry["text"]) != len(entry["labels"]) or len(entry["text"])>512 or len(entry["text"])<200:
        continue  # skip malformed
        print('length not matching')
    if all(label in allowed_labels for label in entry["labels"]):
        examples.append({
            "text": entry["text"],
            "labels": entry["labels"]
        })

print(f"✅ Loaded {len(examples)} usable examples")
max_seq_length = max(len(example["text"]) for example in examples)
print(f"📏 Max sequence length in dataset: {max_seq_length}")
min_seq_length = min(len(example["text"]) for example in examples)
print(f"📏 Min sequence length in dataset: {min_seq_length}")


tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnafm",resume_download=True)
model = RnaFmForTokenPrediction.from_pretrained("multimolecule/rnafm", num_labels=num_labels)
model.config.id2label = id2label
model.config.label2id = label2id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#print(model)

def tokenize_and_align(example, max_length=512):
    # Tokenize with padding/truncation
    encoding = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]
    valid_token_count = attention_mask.sum().item()

    # Map labels to IDs
    label_ids = [label2id[label] for label in example["labels"]]

    # Align label length to actual valid token count (not full max_length)
    if len(label_ids) > valid_token_count:
        label_ids = label_ids[:valid_token_count]
    else:
        label_ids += [-100] * (valid_token_count - len(label_ids))

    # Then pad to match max_length (so shape is still [512])
    label_ids += [-100] * (len(input_ids) - len(label_ids))

    # Final check
    if len(label_ids) != len(input_ids):
        raise ValueError(f"Mismatch after padding: labels={len(label_ids)}, input_ids={len(input_ids)}")

    tokens = {k: v.squeeze().tolist() for k, v in encoding.items()}
    tokens["labels"] = label_ids
    return tokens



train_examples, val_examples = train_test_split(examples, test_size=0.2, random_state=42)
train_dataset = Dataset.from_list(train_examples).map(lambda x: tokenize_and_align(x, max_length=max_seq_length))
val_dataset = Dataset.from_list(val_examples).map(lambda x: tokenize_and_align(x, max_length=max_seq_length))
#for d in train_dataset.select(range(len(train_dataset))):
 #   print(len(d["input_ids"]), len(d["labels"]))

# === Metrics ===
def compute_metrics(p):
    predictions = torch.argmax(torch.tensor(p.predictions), dim=-1)
    labels = torch.tensor(p.label_ids)
    print(f"🔍 Predictions: {predictions.shape}, Labels: {labels.shape}")  #
    true_preds = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        seq_preds = []
        seq_labels = []
        for p_i, l_i in zip(pred_seq, label_seq):
            if l_i != -100:
                seq_preds.append(id2label[int(p_i)])
                seq_labels.append(id2label[int(l_i)])
        true_preds.append(seq_preds)
        true_labels.append(seq_labels)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="./rnafm_token_finetune",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# === Train ===
trainer.train()

# === Inference Function ===
def predict_motifs(sequence):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokens = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}  # move to model's device

    with torch.no_grad():
        outputs = model(**tokens)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
    decoded = [id2label[pred] for i, pred in enumerate(predictions) if tokens["attention_mask"][0][i] == 1]
    return list(sequence), decoded

# === Example Inference ===
seq, predicted_labels = predict_motifs("UAGCUUAUCAGACUGAUGUUG")
print("Sequence:", "".join(seq))
print("Predicted motif labels:", predicted_labels)
