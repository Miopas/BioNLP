import numpy as np

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from feature_generator import FeatureGenerator, Record
from sklearn.model_selection import KFold
import argparse
from cls_utils import *
from sklearn.linear_model import LogisticRegression


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

    param_grid = {'fit_intercept': [True, False],
            'class_weight':['balanced', None],
            'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            }
    clf = LogisticRegression()
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
        clf = LogisticRegression(fit_intercept=best_params['fit_intercept'],
                                class_weight=best_params['class_weight'],
                                solver=best_params['solver']
                                )
        clf.fit(train_data_vectors, train_classes)
        predictions = clf.predict(test_data_vectors)
        test_metrics = get_metrics(predictions, test_classes)

        scores.append(test_metrics)

    print_metrics(scores)

