from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import os
import wandb
wandb.login()
# Step 1: Load the tokenizer and model
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Step 2: Load your dataset — here we use a dummy example
# Replace 'wikitext' with your own text file using `load_dataset('text', data_files={'train': 'your_file.txt'})`
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10]")

# Step 3: Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 4: Create data collator for dynamic masking
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Step 5: Training arguments
training_args = TrainingArguments(
    output_dir="./deberta-mlm-pretrained",
    overwrite_output_dir=True,
    per_device_train_batch_size=16,
    num_train_epochs=1,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
        logging_steps=10,  # How often to log
    report_to="wandb",  # This enables WandB
    run_name="deberta-mlm-run",  # Optional: give a name to your WandB run
    evaluation_strategy="steps",  # Optional: only needed if you want to log eval
    eval_steps=100, 
)

# Step 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 7: Train!
#trainer.train()

# Save final model
#trainer.save_model("./deberta-mlm-final")

