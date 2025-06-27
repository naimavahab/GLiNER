from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from datasets import load_dataset
import numpy as np
import torch
import logging
import time
from datetime import datetime
import wandb

# ====== WandB Setup ======
wandb.login()  # You can also use wandb.init(project="my-project") if needed
wandb.init(project="deberta-mlm-rna", name="mlm-run-1")

# ====== Logging Setup ======
def timestamp():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger("MLM")

# ====== Load Model & Tokenizer ======
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained("deberta_rna_tokeniser")
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# ====== Load & Split Dataset ======
train_dataset = load_dataset('text', data_files='lnc_train1.txt')['train']
eval_dataset = load_dataset('text', data_files='lnc_test1.txt')['train']
# ====== Tokenization ======
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ====== Data Collator ======
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# ====== Custom Metrics ======
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / mask.sum()

    # Masked loss and perplexity
    logits_tensor = torch.tensor(logits[mask], dtype=torch.float32)
    labels_tensor = torch.tensor(labels[mask], dtype=torch.int64)
    loss = torch.nn.functional.cross_entropy(logits_tensor, labels_tensor, reduction='mean').item()
    perplexity = np.exp(loss)
    sample_count = mask.sum()

    logger.info("***** Eval results *****")
    logger.info(f"{timestamp()} | accuracy = {accuracy:.4f}")
    logger.info(f"{timestamp()} | eval_loss = {loss:.4f}")
    logger.info(f"{timestamp()} | perplexity = {perplexity:.4f}")
    logger.info(f"{timestamp()} | eval_samples = {sample_count}")

    # Log to WandB manually (optional)
    wandb.log({
        "accuracy": accuracy,
        "eval_loss": loss,
        "perplexity": perplexity,
        "samples": sample_count,
    })

    return {
        "accuracy": accuracy,
        "eval_loss": loss,
        "perplexity": perplexity,
    }

# ====== Training Arguments ======
training_args = TrainingArguments(
    output_dir="./deberta-mlm-test-run",
    per_device_train_batch_size=8,
    num_train_epochs=10,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="no",
    report_to="wandb",  # ✅ WandB logging
    run_name="deberta-rna-mlm-run",
)

# ====== Trainer ======
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ====== Train & Evaluate ======
trainer.train()
trainer.evaluate()

# ====== Save Model ======
output_dir = "./deberta_mlm_finetuned"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
