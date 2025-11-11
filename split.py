from datasets import load_dataset, concatenate_datasets, DatasetDict

SOURCE_REPO = "Harryxun/GPT-100-pretrain"
TARGET_REPO = "Harryxun/GPT-100-pretrain-splits"
MAX_CHUNKS = 200
TEST_SIZE = 0.01
PUSH = True

chunks = []
for i in range(1, MAX_CHUNKS + 1):
    split_name = f"chunk_{i}"
    try:
        ds = load_dataset(SOURCE_REPO, split=split_name)
        print(f"loaded {split_name}: {len(ds)}")
        chunks.append(ds)
    except Exception:
        break

if not chunks:
    raise SystemExit("No chunk_* splits found.")

full = concatenate_datasets(chunks).shuffle(seed=42)
split = full.train_test_split(test_size=TEST_SIZE, seed=42)

datasets = DatasetDict({"train": split["train"], "test": split["test"]})
print(datasets)

if PUSH:
    datasets.push_to_hub(TARGET_REPO, private=True)
    print(f"Pushed to {TARGET_REPO}")
