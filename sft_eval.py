from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


DATASET_PATH = "Harryxun/stf_run"
SEED = 42

MODEL_PATH = "./llama-pretrained-1/checkpoint-83824"
device = "cuda" 


# load pretrained/sft model/tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()


# example prompt
prompt = """
N = int(input())
total = 0 
X = list( input().split())
for i in X: 
    total +=  (X[i] - N) * (X[i] - N)
print(total)
\n\n### FIXED CODE ###\n\n
"""


inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.1
    )

print("Model:\n", tokenizer.decode(out[0], skip_special_tokens=True))


# # inputs = tokenizer(prompt, return_tensors="pt")
# inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
# outputs = model.generate(inputs)
# print(tokenizer.decode(outputs[0]))