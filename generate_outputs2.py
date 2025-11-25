import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


DATASET_PATH = "Harryxun/GPT-100-sft-dataset-small"
JSONL_PATH = "generations/outputs2.jsonl"

MODEL_PATH = "./llama-sft2/checkpoint-1500"
SEED = 42
device = "cuda" 
DIV = "\n\n### FIXED CODE ###\n\n"


def create_text(example):
    prompt = example["prompt_code"] + DIV
    # train on prompt + answer as one sequence
    return {
        "prompt": prompt,
        "completion": example["correct_code"].strip()
    }


# Load dataset
dataset = load_dataset(DATASET_PATH, split='test')
dataset = dataset.map(create_text)                              # format dataset properly
dataset = dataset.remove_columns(['index', 'task', 'prompt_code'])
print(dataset)


# load pretrained/sft model/tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()


def extract_output(output_str):
    idx = output_str.find(DIV) + len(DIV)
    return output_str[idx:].strip()


def gpu_computation(batch):
    prompts = batch['prompt']
    model_inputs = tokenizer(prompts, padding=True, add_special_tokens=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**model_inputs, max_new_tokens=128)

    output_list = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["output"] = [extract_output(out) for out in output_list]
    return batch


if __name__ == "__main__":
    updated_dataset = dataset.map(
        gpu_computation,
        batched=True,
        batch_size=16,
        # with_rank=True,
        # num_proc=torch.cuda.device_count(),  # one process per GPU
    )

    updated_dataset.to_json(JSONL_PATH)