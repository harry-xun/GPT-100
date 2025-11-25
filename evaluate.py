import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from nltk import edit_distance


OUTPUT1_PATH = 'generations/outputs.jsonl'
OUTPUT2_PATH = 'generations/outputs2.jsonl'
CUTLINE = 0


def prepare_dataloader(path):
    print('Loading data from:', path)
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def format_item1(item):
    return {
        'gold': item['fixed'],
        'pred': item['output']
    }

def format_item2(item):
    return {
        'gold': item['correct_code'],
        'pred': item['output']
    }


# load data
data1 = prepare_dataloader(OUTPUT1_PATH)
data2 = prepare_dataloader(OUTPUT2_PATH)


correct1 = [format_item1(item) for item in data1 if edit_distance(item['output'], item['fixed']) <= CUTLINE]
errors1 = [format_item1(item) for item in data1 if edit_distance(item['output'], item['fixed']) > CUTLINE]
correct2 = [format_item2(item) for item in data2 if edit_distance(item['output'], item['correct_code']) <= CUTLINE]
errors2 = [format_item2(item) for item in data2 if edit_distance(item['output'], item['correct_code']) > CUTLINE]

print("correct1:", len(correct1))
print("errors1:", len(errors1))
print("correct2:", len(correct2))
print("errors2:", len(errors2))

with open('generations/correct1.json', 'w') as f:
    json.dump(correct1, f, indent=4)

with open('generations/errors1.json', 'w') as f:
    json.dump(errors1, f, indent=4)

with open('generations/correct2.json', 'w') as f:
    json.dump(correct2, f, indent=4)

with open('generations/errors2.json', 'w') as f:
    json.dump(errors2, f, indent=4)


# np 1d-array of values
distances1 = [edit_distance(item['output'], item['fixed']) for item in data1]
distances2 = [edit_distance(item['output'], item['correct_code']) for item in data2]

# plot histogram
plt.hist(distances1, weights=np.ones(len(distances1)) / len(distances1))
plt.hist(distances2, weights=np.ones(len(distances2)) / len(distances2))

# y-axis as percentages
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()
