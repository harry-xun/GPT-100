from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DATASET_PATH = "Harryxun/HapyBug-Data"
SEED = 42

# MODEL_PATH = "./sft_output/checkpoint-3050"
MODEL_PATH = "Harryxun/llama-pretrained"
device = "cuda" 

# # Load dataset
# dataset = load_dataset(DATASET_PATH, split='train')
# dataset_test = dataset.train_test_split(test_size=0.2, seed=SEED)['test']

# def format_prompt(example):
#     return example['buggy'] + "\n### FIXED CODE ###\n"


# print("Prompt:\n", format_prompt(dataset_test[0]))
# print("Gold:\n", dataset_test[0]['fixed'])




# load pretrained/sft model/tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()

# prompt = format_prompt(dataset_test[0])
prompt = """\
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create scatter plot with x, y
"""
print("Prompt:\n", prompt)

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