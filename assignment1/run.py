import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from fuzzywuzzy import fuzz
import pandas as pd

class SymptomDetector:
    def __init__(self, symptom_dict_file, neg_trig_file):
        self.threshold = 0.9
        self.window_size = 3
        self.symptom_dict = {}
        with open(symptom_dict_file, 'r') as fr:
            for line in fr:
                cid, keyword = line.strip().split('\t')
                if len(keyword) > 0:
                    self.symptom_dict[keyword.strip().lower()] = cid

        neg_trigger_list = []
        with open(neg_trig_file, 'r') as fr:
            for line in fr:
                neg_trigger_list.append(line.strip())
        self.neg_trig_pattern_str = '|'.join(['({})'.format(t.lower()) for t in neg_trigger_list])

    def process_doc(self, text):
        text = text.lower()
        sentences = sent_tokenize(text)
        output = []

        for sent in sentences:

            # find symptoms
            symptoms = self.find_symptoms(sent)
            if len(symptoms) == 0:
                continue

            # detect negation
            found_cid = {}
            for sym, matched_sym in symptoms:
                cid = self.symptom_dict[sym]
                if cid in found_cid:
                    continue
                found_cid[cid] = 0

                exact_neg_pattern = re.compile(r'(^|\s)({})(\s\w+)?\s{}'.format(self.neg_trig_pattern_str, matched_sym))
                if exact_neg_pattern.search(sent) != None:
                    #print('{}\t{}\t{}\t{}-neg'.format(sent, sym, matched_sym, cid))
                    print('{}\t{}-neg'.format(sent, cid))
                    output.append((cid, '1'))
                else:
                    #print('{}\t{}\t{}\t{}'.format(sent, sym, matched_sym, cid))
                    print('{}\t{}'.format(sent, cid))
                    output.append((cid, '0'))
        return output


    def fuzzy_match(self, k, sent):
        max_score = 0
        max_term = ''
        words = word_tokenize(sent)
        w1 = set(word_tokenize(k))
        for i in range(0, len(words) - self.window_size):
            term = ' '.join(words[i:i+self.window_size])
            score = fuzz.ratio(term, k)/100
            if score > max_score:
                max_score = score
                max_term = term
        return max_term, max_score

    def find_symptoms(self, sent):
        res = []
        for k, v in self.symptom_dict.items():
            matched = ''
            if k in sent: # exact match
                res.append((k, k))
            else:
                term, score = self.fuzzy_match(k, sent)
                if score >= self.threshold:
                    res.append((k, term))
        return res


if __name__ == '__main__':
    symptom_dict_file = './new_dict.tsv'
    neg_trig_file = './neg_trigs.txt'
    infile = './Assignment1GoldStandardSet.xlsx'
    outfile = './result.xlsx'
    detector = SymptomDetector(symptom_dict_file, neg_trig_file)

    df = pd.read_excel(infile)
    new_df = {'ID':[], 'Symptom CUIs':[], 'Negation Flag':[]}
    for index, row in df.iterrows():
        if not pd.isna(row['ID']) and not pd.isna(row['TEXT']):
            new_df['ID'].append(row['ID'])
            output = detector.process_doc(str(row['TEXT']))
            new_df['Symptom CUIs'].append('$$$'.join([x for x, y in output]))
            new_df['Negation Flag'].append('$$$'.join([y for x, y in output]))
    pd.DataFrame(new_df).to_excel(outfile, index=False)
