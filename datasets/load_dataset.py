from datasets import load_dataset

# full_dataset = load_dataset("jtatman/python-code-dataset-500k")

from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset


def any_keyword_in_string(string, keywords):
    for keyword in keywords:
        if keyword in string:
            return True
    return False


def filter_streaming_dataset(dataset, filters):
    filtered_dict = defaultdict(list)
    total = 0
    for sample in tqdm(iter(dataset)):
        total += 1
        if any_keyword_in_string(sample["content"], filters):
            for k, v in sample.items():
                filtered_dict[k].append(v)
    print(f"{len(filtered_dict['content'])/total:.2%} of data after filtering.")
    return Dataset.from_dict(filtered_dict)


def print_size(dataset: Dataset):
    print(f"Dataset size: {dataset.data.nbytes / (1024**3):.2f} GB")


if __name__ == "__main__":
    filters = ["pandas", "tqdm", "matplotlib", "spacy"]
    # example_1 = "import numpy as np"
    # example_2 = "import pandas as pd"

    split = "train"  # "valid"
    filters = ["pandas", "tqdm", "matplotlib", "spacy"]

    data = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)
    filtered_data = filter_streaming_dataset(data, filters)
    print_size(filtered_data)

    filtered_data.push_to_hub("Harryxun/GPT-100-pretrain")