import json
# from datasets import load_dataset, DatasetDict


# DATASET_REPO = "Harryxun/GPT-100-dataset"

# # load dataset
# ds_train = load_dataset(DATASET_REPO, split="train")

# print(ds_train['content'][20])

file_path = f"datasets/FixEval/data/python/jsons/0.json"
with open(file_path, 'r') as f:
    data = json.load(f)

with open('asdf.json', 'w') as f:
    json.dump(data[0][0], f, indent=4)
    json.dump(data[0][1], f, indent=4)

# for pair in data:
#     if len(pair) != 2:
#         print("Data isn't paired")
#         continue
#     assert pair[1]['verdict'] == 'Accepted'

#     data_git['bug_type'].append(pair[0]['verdict'])
#     data_git['buggy'].append(pair[0]['code_tokens'])
#     data_git['fixed'].append(pair[1]['code_tokens'])

