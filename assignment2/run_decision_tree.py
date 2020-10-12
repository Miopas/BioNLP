import numpy as np

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from feature_generator import FeatureGenerator, Record
from sklearn.model_selection import KFold
import argparse
from cls_utils import *
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':

    # Load the data
    f_path = 'pdfalls.csv'
    data = loadDataAsDataFrame(f_path)

    # SPLIT THE DATA (we could use sklearn.model_selection.train_test_split)
    training_set_size = int(0.8 * len(data))
    training_data = data[:training_set_size]
    test_data = data[training_set_size:]

    feature_generator = FeatureGenerator(training_data)
    train_data_vectors, train_classes = feature_generator.transform(training_data)
    test_data_vectors, test_classes = feature_generator.transform(test_data)

    param_grid = {'criterion': ['gini', 'entropy'],
            'max_depth': [2, 5, 10, 15],
            'splitter': ['best', 'random'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 4, 8],
            'max_features' : ['sqrt', 'log2']
            }
    clf = DecisionTreeClassifier()
    clf_search = GridSearchCV(estimator = clf, param_grid=param_grid, scoring='f1_micro', cv=5, n_jobs = -1)
    clf_search.fit(train_data_vectors, train_classes)

    best_params = clf_search.best_params_
    print('best params:{}'.format(best_params))

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
        clf = DecisionTreeClassifier(criterion=best_params['criterion'],
                                    splitter=best_params['splitter'],
                                    max_depth=best_params['max_depth'],
                                    max_features=best_params['max_features'],
                                    min_samples_split=best_params['min_samples_split'],
                                    min_samples_leaf=best_params['min_samples_leaf']
                    )

        clf.fit(train_data_vectors, train_classes)
        predictions = clf.predict(test_data_vectors)
        test_metrics = get_metrics(predictions, test_classes)

        scores.append(test_metrics)

    print_metrics(scores)

