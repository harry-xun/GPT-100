import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.models.utils import setup_chat_format
from peft import LoraConfig
import torch


MODEL_NAME = "Harryxun/llama-pretrained"
DATASET_PATH = "Harryxun/stf_run"
# DATASET_PATH = "Harryxun/sft_time"
SEED = 42
random.seed(SEED)


# Load dataset
dataset = load_dataset(DATASET_PATH, split='train')
# use portion of dataset to speed up training
indices = random.sample(range(len(dataset)), k=50000)
dataset = dataset.select(indices)
# create train/eval splits within original "train" split
dataset = dataset.train_test_split(test_size=0.2, seed=SEED)

# Configure model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=MODEL_NAME).to(device)

# Configure LoRA parameters (identical to ChatKBQA)
# r: rank dimension for LoRA update matrices (smaller = more compression)
rank_dimension = 8
# lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
lora_alpha = 32.0
# lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)
lora_dropout = 0.1

peft_config = LoraConfig(
    modules_to_save= ["embed_tokens", "lm_head"],
    r=rank_dimension,  # Rank dimension - typically between 4-32
    lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
    lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
    bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
    target_modules=["q_proj", "v_proj"],  # Which modules to apply LoRA to
    task_type="CAUSAL_LM",  # Task type for model architecture
)

# Configure trainer
training_args = SFTConfig(
    output_dir="llama-sft",
    dataset_text_field="text",
    max_length=5000,  # max token length: 131072
    # max_steps=1000,
    num_train_epochs=50.0,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    lr_scheduler_type="cosine", 
    learning_rate=5e-5,
    logging_steps=100,
    save_steps=1000,
    eval_strategy="steps",
    eval_steps=500,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,  # LoRA configuration
)

# Start training
trainer.train()
trainer.push_to_hub()
