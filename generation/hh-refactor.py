from datasets import load_dataset, DatasetDict
import random
from huggingface_hub import HfApi, HfFolder


rlhfdataset = load_dataset('Anthropic/hh-rlhf')

# rlhfdataset = rlhfdataset["train"][1]

def merge_selections(example):
    chosen = example['chosen'].split(':')
    reject = example['rejected'].split(':')

    chosen = [part + ':' for part in chosen[:-1]] + [chosen[-1]]
    reject = [part + ':' for part in reject[:-1]] + [reject[-1]]

    min_length = min(len(chosen), len(reject))
    for i in range(-1, -min_length-1, -1):
        if(chosen[i] != reject[i]):
            break

    diff_1 = ''.join(chosen[i:])
    diff_2 = ''.join(reject[i:])

    same = ''.join(chosen[:i])

    if(random.randint(0,1) == 0):
        output = f"{same}\n1. {diff_1}\n2. {diff_2}"
        label = 1
    else:
        output = f"{same}\n1. {diff_2}\n2. {diff_1}"
        label = 2


    return {'text': output, 'label': label}


# res = merge_selections(rlhfdataset)

# print(res['text'])
#
# print(res['label'])
rlhfdataset = rlhfdataset.map(merge_selections)

rlhfdataset = rlhfdataset.remove_columns(['chosen', 'rejected'])

api = HfApi()

rlhfdataset.push_to_hub("davidgaofc/rlhf-transform")
# print(rlhfdataset[0])