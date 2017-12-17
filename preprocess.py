import pandas as pd
import re
import pickle
from tqdm import tqdm


# read table
df = pd.read_csv('data/DIAGNOSES_ICD.csv')
df = df[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']]
df['ICD9_CODE'] = df['ICD9_CODE'].map(lambda x: str(x))

icd_freq = df['ICD9_CODE'].value_counts()

useful_icd = [item for item in icd_freq.keys() if icd_freq[item] >= 5]
useful_icd.sort()

icd2id_dict = {item: key for key, item in enumerate(useful_icd)}
pickle.dump(icd2id_dict, open('data/word_id.p', 'wb'))

# convert it to list
icd_list = []
ez_table = df.values.tolist()

last_id = (0, 0)
last_icd_set = []

for item in tqdm(ez_table):
    item[2] = str(item[2])
    if item[0] != last_id[0] or item[1] != last_id[1]:
        if len(last_icd_set) > 0:
            icd_list.append([last_id, last_icd_set])
        last_id = (item[0], item[1])
        if len(re.findall(r"[A-Z]", item[2])) > 0:
            last_icd_set = []
        else:
            last_icd_set = [item[2]]
    else:
        if len(re.findall(r"[A-Z]", item[2])) == 0:
            last_icd_set.append(item[2])

pickle.dump(icd_list, open('data/icd_list.p', 'wb'))
