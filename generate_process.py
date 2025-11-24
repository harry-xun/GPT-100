import json


GENERATED_PATH = "generations/outputs.jsonl"


def prepare_dataloader():
    print('Loading data from:', GENERATED_PATH)
    with open(GENERATED_PATH, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def find_tgt_id(row):
    return f"{row['problem_id']}_{row['buggy_submission_id']}_{row['problem_id']}_{row['fixed_submission_id']}"


def process_format(data):
    generated = []
    for idx, row in enumerate(data):
        data = {}
        data['tgt_id'] = find_tgt_id(row)
        data['detokenized_src'] = row['buggy']
        data['detokenized_tgt'] = row['fixed']
        data['detokenized_generations'] = row['output']
        generated.append(data)

    with open('generations/formatted.jsonl') as f:
        json.dump(generated, f)
        

if __name__=='__main__':
    data = prepare_dataloader()
    process_format(data)

