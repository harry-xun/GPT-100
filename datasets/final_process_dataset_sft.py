from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi


RUN_DATASET_PATH = "Harryxun/stf_run"
TIME_DATASET_PATH = "Harryxun/sft_time"
SEED = 42


def create_text(example):
    prompt = example["buggy"] + "\n\n### FIXED CODE ###\n\n"
    target = example["fixed"]
    # train on prompt + answer as one sequence
    return {"text": prompt + target}


def update_dataset_and_push(DATASET_PATH):
    # LOAD DATASET
    dataset_train = load_dataset(DATASET_PATH, split='train')
    dataset_test = load_dataset(DATASET_PATH, split='test')

    # process dataset to correct format
    dataset_train_text = dataset_train.map(create_text)
    dataset_test_text = dataset_test.map(create_text)

    # remove unnecessary columns
    if "bug_type" in dataset_train_text.column_names:
        dataset_train_text = dataset_train_text.remove_columns(["bug_type"])
        dataset_test_text = dataset_test_text.remove_columns(["bug_type"])

    # construct final dataset
    dataset_final = DatasetDict({
        "train": dataset_train_text,
        "test": dataset_test_text
    })
    # push to hub
    dataset_final.push_to_hub(DATASET_PATH)
    print(dataset_final)


if __name__ == "__main__":
    update_dataset_and_push(RUN_DATASET_PATH)
    update_dataset_and_push(TIME_DATASET_PATH)