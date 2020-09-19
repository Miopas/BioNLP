ori_file = 'COVID-Twitter-Symptom-Lexicon.txt'
new_file = 'processed_syn_dict.txt'

symptom_dict = {}
with open(ori_file, 'r') as fr:
    for line in fr:
        std_name, cid, keyword = line.strip().split('\t')
        if len(keyword) > 0:
            symptom_dict[keyword.strip().lower()] = cid

with open(new_file, 'r') as fr:
    for line in fr:
        new_term, std_name, neg = line.strip().split('\t')
        if std_name == 'Other':
            symptom_dict[new_term] = 'C0000000'
        #elif neg == '0':
        #    symptom_dict[new_term] = symptom_dict[std_name]

for k, v in symptom_dict.items():
    print('{}\t{}'.format(v, k))
