import numpy as np

from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from feature_generator import FeatureGenerator, Record
from sklearn.model_selection import KFold
import argparse
from cls_utils import *


if __name__ == '__main__':

    # Load the data
    f_path = 'pdfalls.csv'
    data = loadDataAsDataFrame(f_path)

    # SPLIT THE DATA (we could use sklearn.model_selection.train_test_split)
    training_set_size = int(0.8 * len(data))
    training_data = data[:training_set_size]
    test_data = data[training_set_size:]

    # Search hyper-parameters 
    c_params = [1, 2, 4, 8]
    kernel_params = ['linear', 'rbf']

    # K-Fold split
    kf = KFold(n_splits=5)
    kf.get_n_splits(training_data)
    all_scores = []
    for train_index, dev_index in kf.split(training_data):
        ttp_train_data = get_sub(training_data, train_index)
        ttp_dev_data = get_sub(training_data,dev_index)

        feature_generator = FeatureGenerator(ttp_train_data)
        train_data_vectors, train_classes = feature_generator.transform(ttp_train_data)
        dev_data_vectors, dev_classes = feature_generator.transform(ttp_dev_data)
        test_data_vectors, test_classes = feature_generator.transform(test_data)

        scores = []
        models = []
        for c in c_params:
            scores.append([])
            for kernel in kernel_params:
                # TRAIN THE MODEL
                svm_classifier = svm.SVC(C=c, cache_size=200,
                                         coef0=0.0, degree=3, gamma='auto', kernel=kernel, max_iter=-1, probability=True,
                                         random_state=None, shrinking=True, tol=0.001, verbose=False)
                svm_classifier = svm_classifier.fit(train_data_vectors, train_classes)
                predictions = svm_classifier.predict(dev_data_vectors)
                dev_metrics = get_metrics(predictions, dev_classes)

                scores[-1].append(dev_metrics[0])
        all_scores.append(scores)

    all_scores = np.asarray(all_scores) # [k-fold, #c_params, #kernel_params]
    avg_scores = all_scores.mean(0)
    max_idx = np.argmax(avg_scores, axis=1)
    best_c = c_params[max_idx[0]]
    best_kernel = kernel_params[max_idx[1]]
    print('best c:{}, best kernel:{}'.format(best_c, best_kernel))

    # Evaluate the best hyper-parameters with K-Fold split
    kf = KFold(n_splits=5)
    kf.get_n_splits(training_data)
    scores = []
    for train_index, dev_index in kf.split(training_data):

        ttp_train_data = get_sub(training_data, train_index)
        ttp_dev_data = get_sub(training_data,dev_index)

        feature_generator = FeatureGenerator(ttp_train_data)
        train_data_vectors, train_classes = feature_generator.transform(ttp_train_data)
        dev_data_vectors, dev_classes = feature_generator.transform(ttp_dev_data)
        test_data_vectors, test_classes = feature_generator.transform(test_data)

        # TRAIN THE MODEL
        clf = svm.SVC(C=best_c, cache_size=200,
                                         coef0=0.0, degree=3, gamma='auto', kernel=best_kernel, max_iter=-1, probability=True,
                                         random_state=None, shrinking=True, tol=0.001, verbose=False)

        clf.fit(train_data_vectors, train_classes)
        predictions = clf.predict(test_data_vectors)
        test_metrics = get_metrics(predictions, test_classes)

        scores.append(test_metrics)

    print_metrics(scores)

