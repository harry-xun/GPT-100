from datasets import load_dataset


DST_REPO = "Harryxun/GPT-100-dataset-SFT"
DATASET_PATH = "Harryxun/HapyBug-Data"
SEED = 42

# Load dataset
dataset = load_dataset(DATASET_PATH, split='train')
# train_test_split_dataset = dataset.train_test_split(test_size=0.2, seed=SEED)

# print(train_test_split_dataset)
# print(train_test_split_dataset['train'][0])

def create_text(example):
    prompt = example["buggy"] + "\n### FIXED CODE ###\n"
    target = example["fixed"]
    # train on prompt + answer as one sequence
    return {"text": prompt + target}


dataset_text = dataset.map(create_text)
dataset_text = dataset_text.remove_columns(["buggy", "fixed"])
dataset_final = dataset_text.train_test_split(test_size=0.2, seed=SEED)

print(dataset_final)
print(dataset_final['train'][0])
print(type(dataset_final))


dataset_final.push_to_hub(DST_REPO, private=True)