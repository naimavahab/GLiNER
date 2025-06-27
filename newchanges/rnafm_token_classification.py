import torch
from transformers import Trainer, TrainingArguments,AutoTokenizer
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

raw_examples = raw_examples[:10]


# Keep only the labels that your model can handle
allowed_labels = set(label2id.keys())

examples = []
for entry in raw_examples:
    if len(entry["text"]) != len(entry["labels"]):
        continue  # skip malformed
    if all(label in allowed_labels for label in entry["labels"]):
        examples.append({
            "text": entry["text"],
            "labels": entry["labels"]
        })

print(f"✅ Loaded {len(examples)} usable examples")
max_seq_length = max(len(example["text"]) for example in examples)
print(f"📏 Max sequence length in dataset: {max_seq_length}")
print('example dataset',examples[0])

# === Tokenizer & Model ===
tokenizer = AutoTokenizer.from_pretrained("./rnafm-fast-tokenizer") #RnaTokenizer.from_pretrained("multimolecule/rnafm",resume_download=True)
model = RnaFmForTokenPrediction.from_pretrained("multimolecule/rnafm",id2label=id2label,label2id=label2id)
#model.config.id2label = id2label
#model.config.label2id = label2id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# === Preprocessing: Tokenize & Align Labels ===

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
#    print('wordids',word_ids)
 #   print('labels',labels)
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            # Special token
            print('special token', word_id)
            new_labels.append(-100)
        else:
            # Same word as previous token
            label =  labels[word_id]
            new_labels.append(label)

    return new_labels



def tokenize_and_align(example, max_length=512):
    tokens = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
   
    all_labels = example["labels"]
    print('inside tokenize and laign')
 #   print(tokens)
  #  print(all_labels)
    word_ids = tokens.word_ids()
    print(len(word_ids))
    print(len(all_labels))
    new_labels =align_labels_with_tokens(all_labels, word_ids)
#    for i, label in enumerate(all_labels):
 #       word_id = tokens.word_ids(i)
   # print('Here word_id')
   #     print(word_id)
    #print(label)
  #      new_labels.append(align_labels_with_tokens(label, word_id))

    tokens["labels"] = new_labels

    return tokens


train_examples, val_examples = train_test_split(examples, test_size=0.2, random_state=42)
print(train_examples)
print(Dataset.from_list(train_examples))
tokenized_datasets = Dataset.from_list(train_examples).map(
    tokenize_and_align,
#    batched=True,
#    remove_columns=train_examples["train"].column_names,
)
#train_dataset = Dataset.from_list(train_examples).map(lambda x: tokenize_and_align(x, max_length=max_seq_length))
#val_dataset = Dataset.from_list(val_examples).map(lambda x: tokenize_and_align(x, max_length=max_seq_length))
#print(train_dataset[0])

# === Metrics ===
def compute_metrics(p):
    predictions = torch.argmax(torch.tensor(p.predictions), dim=-1)
    labels = torch.tensor(p.label_ids)

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
#trainer.train()

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
