from prima.privacy import PriMa
from datasets import load_dataset

dataset = load_dataset('davidgaofc/RM_inout')
prima = PriMa()
flag = 0
def process_column(row):
    try:
        processed = prima.augment(row['Text'])
    except:
        print(row['Text'])
        processed = "skip"
        flag = 1
    return {'Text': processed, 'Label': row['Label']}

processed_dataset = dataset.map(process_column)

if flag == 0:
    processed_dataset.push_to_hub('davidgaofc/PRIMA_inout')
