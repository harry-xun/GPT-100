from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json


DST_REPO = "Harryxun/GPT-100-dataset-SFT"
DATASET_PATH = "Harryxun/HapyBug-Data"
SEED = 42

# Load dataset
# dataset = load_dataset(DATASET_PATH, split='train')

# def create_text(example):
#     prompt = example["buggy"] + "\n### FIXED CODE ###\n"
#     target = example["fixed"]
#     # train on prompt + answer as one sequence
#     return {"text": prompt + target}


# dataset_text = dataset.map(create_text)
# dataset_text = dataset_text.remove_columns(["buggy", "fixed"])
# dataset_final = dataset_text.train_test_split(test_size=0.2, seed=SEED)

# print(dataset_final)
# print(dataset_final['train'][0])
# print(type(dataset_final))


# dataset_final.push_to_hub(DST_REPO, private=True)

with open('datasets/HaPy-Bug/annotated_data/A_1_24.json', 'r') as file:
    data_raw = json.load(file)
# print(data_raw[0])
print(json.dumps(data_raw[0], indent=4))
print(data_raw[0]['data']['gitCommits'][0]['diff'])

# dataset = load_dataset(DATASET_PATH, split='train')
# print(dataset['buggy'])
# print(dataset['fixed'])