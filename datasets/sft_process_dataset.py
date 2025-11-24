import os
import json
import csv
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sft_split_dataset import problem_split


HF_REPO_ID = "Harryxun/GPT-100-sft-dataset"
SEED = 42

ROOT = Path(__file__).resolve().parent / "FixEval"
JSON_DIR = ROOT / "data" / "python" / "jsons"
PROBLEM_LIST = ROOT / "src" / "problem_list.csv"

BUG_VERDICTS = {"Runtime Error"}
AC_VERDICT = "Accepted"


def get_usable_pids():
    pid2info = {}
    with PROBLEM_LIST.open() as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("dataset") != "AtCoder":
                continue
            pid = row["id"]
            name = row["name"]
            pid2info[pid] = (None, name)
    return pid2info


def load_pairs(pid2info):
    b_code = []
    f_code = []
    b_sid = []
    f_sid = []
    pids = []
    langs = []
    b_ver = []

    files = sorted(JSON_DIR.glob("*.json"))

    for jf in files:
        with jf.open() as f:
            try:
                data = json.load(f)
            except Exception:
                continue

        for g in data:
            if isinstance(g, dict):
                subs = [g]
            elif isinstance(g, list):
                subs = g
            else:
                continue

            subs = [s for s in subs if str(s.get("lang", "")).lower() == "python"]
            if not subs:
                continue

            pid = subs[0].get("problem_id")
            if pid not in pid2info:
                continue

            ac = [s for s in subs if s.get("verdict") == AC_VERDICT]
            bg = [s for s in subs if s.get("verdict") in BUG_VERDICTS]

            if not ac or not bg:
                continue

            fixed = ac[0]
            f_c = fixed.get("code_tokens", "")
            if isinstance(f_c, list):
                f_c = " ".join(f_c)
            if not f_c:
                continue

            f_id = fixed.get("submission_id")
            lang = fixed.get("lang", "python")

            for s in bg:
                c = s.get("code_tokens", "")
                if isinstance(c, list):
                    c = " ".join(c)
                if not c:
                    continue

                b_code.append(c)
                f_code.append(f_c)
                b_sid.append(s.get("submission_id"))
                f_sid.append(f_id)
                pids.append(pid)
                langs.append(lang)
                b_ver.append(s.get("verdict"))
 
    ds = Dataset.from_dict(
        {
            "buggy": b_code,
            "fixed": f_code,
            "buggy_submission_id": b_sid,
            "fixed_submission_id": f_sid,
            "problem_id": pids,
            "lang": langs,
            "buggy_verdict": b_ver,
        }
    )
    return ds


def split_dataset(ds):
    test_set = problem_split()  # set
    
    train_jsonl = []
    test_jsonl = []

    for ex in ds:
        if ex['problem_id'] in test_set:
            test_jsonl.append(ex)
        else:
            train_jsonl.append(ex)
    ds_train = Dataset.from_pandas(pd.DataFrame(data=train_jsonl))
    ds_test = Dataset.from_pandas(pd.DataFrame(data=test_jsonl))
    ds_dict = DatasetDict({
        'train': ds_train,
        'test': ds_test,
    })
    return ds_dict


def push(ds_dict):
    api = HfApi()
    try:
        api.create_repo(HF_REPO_ID, private=False)
    except Exception:
        pass
    ds_dict["train"].push_to_hub(HF_REPO_ID, split="train")
    ds_dict["test"].push_to_hub(HF_REPO_ID, split="test")


def main():
    # delete_files_in_repo()
    pid2info = get_usable_pids()
    ds = load_pairs(pid2info)

    ds_dict = split_dataset(ds)
    push(ds_dict)


if __name__ == "__main__":
    main()
