import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from fuzzywuzzy import fuzz
import pandas as pd
import string
import sys

EXACT_ENBALE = int(sys.argv[1])
FUZZY_ENBALE = int(sys.argv[2])
EXTRA_ENBALE = int(sys.argv[3])

class SymptomDetector:
    def __init__(self, symptom_dict_file, neg_trig_file, extra_dict_file):
        if FUZZY_ENBALE:
            self.threshold = 0.8
        else:
            self.threshold = 999

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

        self.extra_dict = []
        with open(extra_dict_file, 'r') as fr:
            for line in fr:
                cid, ptn = line.strip().split('\t')
                self.extra_dict.append((re.compile(ptn), cid))

        if not EXTRA_ENBALE:
            self.extra_dict = []

    def detect_neg(self, sent, matched_sym):
        exact_neg_pattern = re.compile(r'(^|\s)({})(\s\w+\s\w+)?\s{}'.format(self.neg_trig_pattern_str, matched_sym))

        if exact_neg_pattern.search(sent) != None:
            return 1
        else:
            return 0

    def process_doc(self, text, qid):
        text = text.lower()
        sentences = sent_tokenize(text)
        output = []

        for sent in sentences:
            # preprocess sent
            sent = re.sub('[{}]'.format(string.punctuation), '', sent)

            # find symptoms
            symptoms, fuzzy_flags = self.find_symptoms(sent)

            found_cid = {}
            if len(symptoms) == 0:
                for ptn, cid in self.extra_dict:
                    matched = ptn.findall(sent)
                    if len(matched) > 0:
                        matched.sort(key=len)
                        matched_sym = matched[0]
                        print('EXTRA:{}\t{}\t{}\t{}'.format(sent, matched_sym, qid, cid))
                        if self.detect_neg(sent, matched_sym):
                            output.append((cid, '1'))
                        else:
                            output.append((cid, '0'))
            else:
                for (sym, matched_sym), fuzzy_flag in zip(symptoms, fuzzy_flags):
                    cid = self.symptom_dict[sym]
                    if cid in found_cid:
                        continue
                    found_cid[cid] = 0

                    print('FUZZY{}:{}\t{}\t{}\t{}'.format(fuzzy_flag, sent, matched_sym, qid, cid))
                    if self.detect_neg(sent, matched_sym):
                        output.append((cid, '1'))
                    else:
                        output.append((cid, '0'))
        return output


    def fuzzy_match(self, k, sent):
        max_score = 0
        max_term = ''
        words = word_tokenize(sent)
        #ws = len(word_tokenize(k))
        ws = self.window_size
        for i in range(0, len(words) - ws):
            term = ' '.join(words[i:i+ws])
            score = fuzz.ratio(term, k)/100
            if score > max_score:
                max_score = score
                max_term = term
        return max_term, max_score

    def find_symptoms(self, sent):
        res = []
        fuzzy_flags = []
        for k, v in self.symptom_dict.items():
            if k in sent and EXACT_ENBALE: # exact match
            #if 0: # exact match
                res.append((k, k))
                fuzzy_flags.append(0)
            else:
                term, score = self.fuzzy_match(k, sent)
                if score >= self.threshold:
                    res.append((k, term))
                    fuzzy_flags.append(1)
        return res, fuzzy_flags


if __name__ == '__main__':
    symptom_dict_file = './COVID-Twitter-Symptom-Lexicon.tsv'
    neg_trig_file = './neg_trigs.txt'
    extra_dict_file = './keywords.tsv'
    infile = './Assignment1GoldStandardSet.xlsx'
    outfile = './result.xlsx'
    #infile = 's10_annotated.xlsx'
    #outfile = 's10_annotated_result.xlsx'
    detector = SymptomDetector(symptom_dict_file, neg_trig_file, extra_dict_file)

    df = pd.read_excel(infile)
    new_df = {'ID':[], 'Symptom CUIs':[], 'Negation Flag':[]}
    for index, row in df.iterrows():
        if not pd.isna(row['ID']) and not pd.isna(row['TEXT']):
            new_df['ID'].append(row['ID'])
            output = detector.process_doc(str(row['TEXT']), row['ID'])
            new_df['Symptom CUIs'].append('$$$' + '$$$'.join([x for x, y in output]) + '$$$')
            new_df['Negation Flag'].append('$$$' +'$$$'.join([y for x, y in output]) + '$$$')
    pd.DataFrame(new_df).to_excel(outfile, index=False)
