from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


MODEL_NAME = "meta-llama/Llama-3.2-1B"
DATASET_REPO = "Harryxun/GPT-100-dataset"
BATCH_LENGTH = 128


# load dataset
ds_train = load_dataset(DATASET_REPO, split="train")
ds_test = load_dataset(DATASET_REPO, split="test")


# potentially select subset
pretrain_dataset = DatasetDict(
    {
        "train": ds_train,       # .shuffle().select(range(50000)),
        "test": ds_test,         # .shuffle().select(range(1000))
    }
)


# tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=BATCH_LENGTH,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == BATCH_LENGTH:
            # input_ids = input_ids + [eos_id]  # append EOS after each example
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

tokenized_datasets = pretrain_dataset.map(
    tokenize, batched=True, remove_columns=pretrain_dataset["train"].column_names
)


# initializing model
config = AutoConfig.from_pretrained(
    MODEL_NAME,
    vocab_size=len(tokenizer),
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = LlamaForCausalLM(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"Llama size: {model_size/1000**2:.1f}M parameters")


# running example
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)  # running for causal, not masked

# training
args = TrainingArguments(
    output_dir="llama-pretrained",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=5_000,          # change back to 5000
    logging_steps=5_000,       # change back to 5000
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()
# trainer.push_to_hub()