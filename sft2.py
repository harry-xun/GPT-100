import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.models.utils import setup_chat_format
from peft import LoraConfig
import torch


MODEL_PATH = "./llama-pretrained-1/checkpoint-83824"
DATASET_PATH = "Harryxun/GPT-100-sft-dataset-small"
SEED = 42
random.seed(SEED)


def create_text(example):
    prompt = example["prompt_code"] + "\n\n### FIXED CODE ###\n\n"
    # train on prompt + answer as one sequence
    return {
        "prompt": prompt,
        "completion": example["correct_code"]
    }


# Load dataset
dataset = load_dataset(DATASET_PATH, split='train')

dataset = dataset.map(create_text)                              # format dataset properly
dataset = dataset.train_test_split(test_size=0.2, seed=SEED)    # create train/eval splits within original "train" split
dataset = dataset.remove_columns(['full', 'correct_code', 'index', 'task', 'prompt_code'])
print(dataset)


# Configure model and tokenizer to match pretraining
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=MODEL_PATH).to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token


# Configure LoRA parameters (identical to ChatKBQA)
rank_dimension = 16      # r: rank dimension for LoRA update matrices (smaller = more compression)
lora_alpha = 32.0       # lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
lora_dropout = 0.3      # lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)

peft_config = LoraConfig(
    # modules_to_save= ["embed_tokens", "lm_head"],
    r=rank_dimension,  # Rank dimension - typically between 4-32
    lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
    lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
    bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
    target_modules=["q_proj", "v_proj"],  # Which modules to apply LoRA to
    task_type="CAUSAL_LM",  # Task type for model architecture
)


# Trainer config
training_args = SFTConfig(
    output_dir="llama-sft2",
    # dataset_text_field="text",
    max_length=5000,  # max token length: 131072
    # max_steps=1000,
    num_train_epochs=50.0,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    lr_scheduler_type="cosine", 
    learning_rate=5e-4,
    logging_steps=100,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
    eos_token=tokenizer.eos_token,
    pad_token=tokenizer.pad_token,
    packing=False,
)


# Trainer initialization
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,  # LoRA config
    processing_class=tokenizer,
)


# Start training
trainer.train()
trainer.push_to_hub()
