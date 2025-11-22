import json
import pandas as pd
from datasets import Dataset, load_dataset


SEED = 42
# df = pd.read_csv('kaggle_dataset.csv')
# dataset_kaggle = Dataset.from_pandas(df)
# dataset_hf = load_dataset('Muennighoff/python-bugs')


##### FixEval ####
def fixeval_filter_and_push(repo_id_run, repo_id_time):

    MAX_FILE_NUM = 115
    data_git = {'bug_type': [], 'buggy': [], 'fixed': []}

    for num in range(MAX_FILE_NUM + 1):
        file_path = f"FixEval/data/python/jsons/{num}.json"
        with open(file_path, 'r') as f:
            data = json.load(f)

        for pair in data:
            if len(pair) != 2:
                print("Data isn't paired")
                continue
            assert pair[1]['verdict'] == 'Accepted'

            data_git['bug_type'].append(pair[0]['verdict'])
            data_git['buggy'].append(pair[0]['code_tokens'])
            data_git['fixed'].append(pair[1]['code_tokens'])

    dataset_fixeval = Dataset.from_dict(data_git)
    print(dataset_fixeval)

    # bug_types = set(dataset_fixeval['bug_type'])
    # for bug in bug_types:
    #     print("BUG:", bug)
    #     print(dataset_fixeval.filter(lambda example: example["bug_type"] == bug))
    # bugs: Runtime Error, Time Limit Exceeded

    dataset_runtime = dataset_fixeval.filter(lambda example: example["bug_type"] == 'Runtime Error')
    dataset_runtime = dataset_runtime.train_test_split(test_size=0.2, seed=SEED)
    dataset_runtime.push_to_hub(repo_id_run)

    dataset_timelim = dataset_fixeval.filter(lambda example: example["bug_type"] == 'Time Limit Exceeded')
    dataset_timelim = dataset_timelim.train_test_split(test_size=0.2, seed=SEED)
    dataset_timelim.push_to_hub(repo_id_time)
    

if __name__ == "__main__":
    repo_id_run = 'Harryxun/stf_run'
    repo_id_time = 'Harryxun/sft_time'
    fixeval_filter_and_push(repo_id_run, repo_id_time)

