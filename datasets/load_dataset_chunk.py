from datasets import load_dataset, Dataset
from tqdm import tqdm
from collections import defaultdict
import os

def any_keyword_in_string(string, keywords):
    return any(k in string for k in keywords)

def stream_filter_and_push(dataset, filters, repo_id, chunk_size=50000):
    total = 0
    kept = 0
    batch = defaultdict(list)
    batch_idx = 0

    for sample in tqdm(dataset):
        total += 1
        if any_keyword_in_string(sample["content"], filters):
            for k, v in sample.items():
                batch[k].append(v)
            kept += 1

        if len(batch["content"]) >= chunk_size:
            batch_idx += 1
            print(f"Pushing batch {batch_idx} with {len(batch['content'])} samples...")
            temp_ds = Dataset.from_dict(batch)
            temp_ds.push_to_hub(repo_id, split=f"chunk_{batch_idx}", private=True)
            batch = defaultdict(list) 

    if batch["content"]:
        batch_idx += 1
        print(f"Pushing final batch {batch_idx} with {len(batch['content'])} samples...")
        temp_ds = Dataset.from_dict(batch)
        temp_ds.push_to_hub(repo_id, split=f"chunk_{batch_idx}", private=True)

    print(f"Done. Filtered {kept}/{total} ({kept/total:.2%}) samples.")

if __name__ == "__main__":
    filters = ["pandas", "tqdm", "matplotlib", "spacy"]
    split = "train"

    data = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)

    stream_filter_and_push(
        data,
        filters,
        repo_id="Harryxun/GPT-100-pretrain",  
        chunk_size=50000         
    )
