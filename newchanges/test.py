'''
from multimolecule import RnaTokenizer, RiNALMoModel


tokenizer = RnaTokenizer.from_pretrained("multimolecule/rinalmo")
model = RiNALMoModel.from_pretrained("multimolecule/rinalmo")

text = "UAGCUUAUCAGACUGAUGUUG"
input = tokenizer(text, return_tensors="pt")
print(input)
output = model(**input)
print(output)

import torch
from multimolecule import RnaTokenizer, RiNALMoForSequencePrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/rinalmo")
model = RiNALMoForSequencePrediction.from_pretrained("multimolecule/rinalmo")

text = "UAGCUUAUCAGACUGAUGUUG"
input = tokenizer(text, return_tensors="pt")
label = torch.tensor([1])

output = model(**input, labels=label)
print(label)
print(output)
'''
import torch
from multimolecule import RnaTokenizer, RiNALMoForTokenPrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/rinalmo")
model = RiNALMoForTokenPrediction.from_pretrained("multimolecule/rinalmo",num_labels=2)

text = "UAGCUUAUCAGACUGAUGUUG"
inputs = tokenizer(text, return_tensors="pt")
print({k: v.shape for k, v in inputs.items()})
labels = torch.randint(2, (len(text), ))

#pad_len = inputs["input_ids"].size(1) - labels.size(1)  # 23 - 21 = 2
#labels_padded = torch.cat([labels, torch.full((1, pad_len), -100)], dim=1)  # shape (1,23) # shape: [1, 21]
print("Input IDs shape:", inputs["input_ids"].shape)
#print("Labels shape:", labels_padded.shape)
print("Labels unique values:", torch.unique(labels))

output = model(**inputs, labels=labels)
