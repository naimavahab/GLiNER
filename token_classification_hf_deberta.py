import torch
from transformers import Trainer, TrainingArguments,AutoTokenizer
from multimolecule import RnaTokenizer, RnaFmForTokenPrediction
from datasets import Dataset
import json,numpy
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import evaluate
from transformers import AutoModelForTokenClassification,DataCollatorForTokenClassification
# === Label Setup ===

'''
Data processing
'''

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
print(id2label)
print(num_labels)

with open("rna_substructure_examples.json", "r") as f:
    raw_examples = json.load(f)
raw_examples_ = [
  {
    "Name": "bpRNA_CRW_31852",
    "text": "GCG1233",
    "labels": [
        "O",
      "O",
      "MULTILOOP",
      "MULTILOOP",    
      "HAIRPIN",
      "HAIRPIN",
      "HAIRPIN"
    ]
  }]
for entry in raw_examples:
    entry['sequence'] = list(entry['text'])
#raw_examples = raw_examples[:5]
train_examples, val_examples = train_test_split(raw_examples, test_size=0.2, random_state=42)
train_examples = Dataset.from_list(train_examples)
val_examples = Dataset.from_list(val_examples)
#tokenizer1 = AutoTokenizer.from_pretrained("./deberta_pretrained") 
tokenizer = AutoTokenizer.from_pretrained("./deberta_pretrained")


'''
if the input ids <6 then append -100 to the labels
'''
def align_labels_with_tokens(tokens,example):
    new_labels = []
    label_ptr = 0
    graph = tokens['input_ids']
    labels = example['labels']
    for value in graph:
        if value in [1, 2, 3,5]:
            new_labels.append(-100)
        else:
            if label_ptr < len(labels):      
                new_labels.append(label2id[labels[label_ptr]])
                label_ptr += 1
            else:
                new_labels.append(-100)  # Just in case labels run out
  #  new_labels = new_labels[1:-1] #due to error in the loss funciton due to input and output size mismatch
    #dropping the cls and eos labels ; the model may be truncating the cls and eos tokens during training
    tokens['labels'] = new_labels
    return tokens


def tokenize_and_align(example):
  
    tokens = tokenizer(
        example["text"],
        truncation=True,max_length=16  #,padding='max_length', max_length=15  
    )   
    tokens = align_labels_with_tokens(tokens,example)
    return tokens

train_datasets = train_examples.map(
    tokenize_and_align
#    remove_columns=train_examples["train"].column_names,
)
val_datasets = val_examples.map(
    tokenize_and_align
#    remove_columns=train_examples["train"].column_names,
)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
print(train_datasets)
for i in train_datasets:
    if len(i['input_ids']) == len(i['labels']):
        print('matching')
    else:
        print('not matching')

'''
Evaluation
'''
metric = evaluate.load("seqeval")

import numpy as np


def compute_metrics_(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    print(labels)
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    print("logits.shape:", logits.shape)
    print("labels.shape:", labels.shape)
    print("predictions.shape:", predictions.shape)

    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    labels = labels.tolist()
    predictions = predictions.tolist()

    true_labels = []
    true_predictions = []

    for pred_seq, label_seq in zip(predictions, labels):
        seq_preds = []
        seq_labels = []
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                seq_preds.append(id2label[pred])
                seq_labels.append(id2label[label])
        true_predictions.append(seq_preds)
        true_labels.append(seq_labels)

    # Debug: check shape
    print("Sample true_predictions:", true_predictions[:1])
    print("Sample true_labels:", true_labels[:1])
    assert all(isinstance(x, list) for x in true_predictions), "Predictions not list of lists"
    assert all(isinstance(x, list) for x in true_labels), "Labels not list of lists"

    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

class DebugTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):       
  
        labels = inputs.get("labels")
       # labels = labels[:, 1:-1]
     #   inputs["labels"] = inputs["labels"][:, 1:-1] #this was done to resolve shape mismatch in the tensor. may be model
        #implementaiton not correct;trimmed the lables eos and cls tokens.
        outputs = model(**inputs)
        logits = outputs.get("logits")
        print("Logits shape:", logits.shape)  # e.g., [batch_size, num_labels]
        print("Labels shape:", labels.shape)
        print("Unique label values:", torch.unique(labels))

        num_labels = model.module.config.num_labels
        print("Valid label range: 0 to", num_labels - 1)
        print("Min label:", labels.min().item(), "Max label:", labels.max().item())

     #   loss_fct = torch.nn.CrossEntropyLoss()
       # loss = loss_fct(logits.reshape(-1, model.module.config.num_labels), labels.reshape(-1))
      #  loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
# logits: [batch_size, seq_len, num_labels]
# labels: [batch_size, seq_len]
        #loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        loss = loss_fct(logits.reshape(-1, model.module.config.num_labels), labels.reshape(-1))

        return (loss, outputs) if return_outputs else loss
        

model = AutoModelForTokenClassification.from_pretrained(
    "./deberta_pretrained",num_labels = 7
    #id2label=id2label,
   # label2id=label2id,
)
print(model.config.num_labels)


args = TrainingArguments(
    "bert-finetuned-ner",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=10,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8
)
trainer = DebugTrainer(
    model=model,
    args=args,
    train_dataset=train_datasets,
    eval_dataset=val_datasets,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model('deberta-finetuned-for-token-classification')
