import pickle
from collections import Counter
from tqdm import tqdm

word_id = pickle.load(open('data/word_id.p', 'rb'))
unique_word = word_id.keys()
unknown_id = len(unique_word)

icd_list = pickle.load(open('data/icd_list.p', 'rb'))
icd_pair_counter = Counter()

for item in tqdm(icd_list):
    list_ins = item[1]

    for icd_1 in list_ins:
        id_1 = word_id[icd_1] if icd_1 in unique_word else unknown_id
        for icd_2 in list_ins:
            id_2 = word_id[icd_2] if icd_2 in unique_word else unknown_id
            if icd_1 != icd_2:
                icd_pair_counter.update([(id_1, id_2)])

# reshape counters
data_table = []
for item in icd_pair_counter.keys():
    data_table.append((item[0], item[1], icd_pair_counter[item]))

pickle.dump(data_table, open('data/word2vec_table.p', 'wb'))