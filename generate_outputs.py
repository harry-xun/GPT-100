import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


DATASET_PATH = "Harryxun/GPT-100-sft-dataset"
JSONL_PATH = "generations/outputs.jsonl"

MODEL_PATH = "Harryxun/llama-sft"
SEED = 42
device = "cuda" 
DIV = "\n\n### FIXED CODE ###\n\n"


def create_text(example):
    return {"text": example["buggy"] + DIV}


# Load dataset subset --> evaluating all with FixEval test suite would take too long
dataset = load_dataset(DATASET_PATH, split='test').shuffle(seed=SEED).select(range(1000))
dataset = dataset.map(create_text)


# load pretrained/sft model/tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()


def extract_output(output_str):
    idx = output_str.find(DIV) + len(DIV)
    return output_str[idx:]


def gpu_computation(batch):
    prompts = batch['text']
    model_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**model_inputs, max_new_tokens=128)

    batch["output"] = extract_output(tokenizer.batch_decode(outputs, skip_special_tokens=True))
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