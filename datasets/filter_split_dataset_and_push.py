from datasets import load_dataset, DatasetDict

SRC_REPO = "Harryxun/GPT-100-rawdataset"       
DST_REPO = "Harryxun/GPT-100-dataset"
SEED = 42

dataset = load_dataset(SRC_REPO, split="train")

splits = dataset.train_test_split(test_size=0.2, seed=SEED, shuffle=True)
dataset = DatasetDict({
    "train": splits["train"],
    "test":  splits["test"],
})
dataset.push_to_hub(DST_REPO, private=True)