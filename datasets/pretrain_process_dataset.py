from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from collections import defaultdict
import os


SRC_REPO = "Harryxun/GPT-100-rawdataset"       
DST_REPO = "Harryxun/GPT-100-dataset"
SEED = 42


def any_keyword_in_string(string, keywords):
    return any(k in string for k in keywords)


def stream_filter_and_push(dataset, filters, repo_id, split_name="train"):
    total = 0
    kept = 0
    batch = defaultdict(list)

    for sample in tqdm(dataset):
        total += 1
        if any_keyword_in_string(sample["content"], filters):
            for k, v in sample.items():
                batch[k].append(v)
            kept += 1

    if batch["content"]:
        print(f"Pushing ALL data with {len(batch['content'])} samples to split '{split_name}'...")
        temp_ds = Dataset.from_dict(batch)
        temp_ds.push_to_hub(repo_id, split=split_name, private=True)

    if total > 0:
        print(f"Done. Filtered {kept}/{total} ({kept/total:.2%}) samples.")
    else:
        print("Done. No samples seen.")


def split_train_test(dataset):
    splits = dataset.train_test_split(test_size=0.2, seed=SEED, shuffle=True)
    dataset = DatasetDict({
        "train": splits["train"],
        "test":  splits["test"],
    })
    dataset.push_to_hub(DST_REPO, private=True)


if __name__ == "__main__":
    filters = ["pandas", "keras", "matplotlib", "spacy"] 
    split = "train"

    # load raw dataset (only train - still excessive examples)
    data = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)

    # filter and (temporarily) push
    stream_filter_and_push(
        data,
        filters,
        repo_id=SRC_REPO,
        split_name=split
    )
    
    # split dataset into train/test
    raw_dataset = load_dataset(SRC_REPO, split="train")
    split_train_test(raw_dataset)