from datasets import load_dataset, DatasetDict, Dataset
import random
from huggingface_hub import HfApi, HfFolder
import pandas as pd


dataset = load_dataset('davidgaofc/MedQuad_split')


sampled_a = dataset['SFT_train2'].shuffle().select(range(820))['Question']
sampled_b = dataset['Shadow_oos']['Question']
# sampled_a = dataset['SFT_train1'].shuffle().select(range(820))['Question']
# sampled_b = dataset['RM_oos']['Question']


data_a = pd.DataFrame(sampled_a, columns=['Question'])
data_a['label'] = 0
data_b = pd.DataFrame(sampled_b, columns=['Question'])
data_b['label'] = 1

# Concatenate and save to CSV
combined_data = pd.concat([data_a, data_b])
# combined_data.to_csv("RM.csv", index=False)
combined_data.to_csv("Shadow.csv", index=False)

new_dataset = Dataset.from_csv("../data/RM.csv")


new_dataset.push_to_hub("davidgaofc/Shadow_prompts")
# new_dataset.push_to_hub("davidgaofc/RM_prompts")