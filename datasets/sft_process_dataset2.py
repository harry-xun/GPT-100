from huggingface_hub import HfApi
from datasets import load_dataset, Dataset, DatasetDict


DATASET_ID = "Muennighoff/python-bugs"
HF_REPO_ID = "Harryxun/GPT-100-sft-dataset-small"
BUG_TYPE = 'var-misuse'  # or bin-op
SEED = 42


def push(ds_dict):
    api = HfApi()
    try:
        api.create_repo(HF_REPO_ID, private=False)
    except Exception:
        pass
    ds_dict["train"].push_to_hub(HF_REPO_ID, split="train")
    ds_dict["test"].push_to_hub(HF_REPO_ID, split="test")


def main():
    dataset = load_dataset(DATASET_ID, split='train')
    filtered_dataset = dataset.filter(lambda example: example["task"] == BUG_TYPE)
    final_dataset = filtered_dataset.train_test_split(test_size=0.2, seed=SEED, shuffle=True)
    push(final_dataset)


if __name__ == "__main__":
    main()