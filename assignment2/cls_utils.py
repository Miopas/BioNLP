import numpy as np
import pandas as pd
from feature_generator import FeatureGenerator, Record
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from scipy import stats
from itertools import combinations

def loadDataAsDataFrame(f_path):
    '''
        Given a path, loads a data set and puts it into a dataframe
        - simplified mechanism
    '''
    df = pd.read_csv(f_path) # sorted by record_id
    data = []
    last_id = None
    prev_day = 0
    for i, row in df.iterrows():
        uid = row['record_id']
        day = row['fall_study_day']

        if last_id != uid:
            prev_day = 0

        data.append(Record(row, prev_day))
        prev_day = day
        last_id = uid
    return data


def get_voting(cls_models, data_vectors):
    if len(cls_models) == 1:
        clf = cls_models[0]
        return clf.predict(data_vectors) 
    predictions = np.array([clf.predict(data_vectors) for clf in cls_models])
    n, m = predictions.shape
    modes = stats.mode(predictions, axis=0).mode
    predictions = modes.reshape(m,).tolist()
    return predictions


def get_sub(arr, indices):
    return [arr[i] for i in indices]


def get_metrics(y_pred, y_true):
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_pred, y_true)
    return (f1_micro, f1_macro, acc)


def print_metrics(scores):
    print('f1_micro\tf1_macro\tacc')
    print('metrics:{0:.3f}\t{1:.3f}\t{2:.3f}'.format(sum([x[0] for x in scores])/len(scores), sum([x[1] for
                                x in scores])/len(scores), sum([x[2] for x in scores])/len(scores)))

def get_combs(n):
    all_idx = [i for i in range(n)]
    res = []
    for i in range(n):
        for x in combinations(all_idx, i+1):
            res.append(list(x))
    return res

if __name__ == '__main__':
    print(get_combs(5))
