from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from feature_generator import FeatureGenerator, Record
from sklearn.model_selection import KFold
import argparse
from cls_utils import *

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':

    # Load the data
    f_path = 'pdfalls.csv'
    data = loadDataAsDataFrame(f_path)

    # SPLIT THE DATA (we could use sklearn.model_selection.train_test_split)
    training_set_size = int(0.8 * len(data))
    training_data = data[:training_set_size]
    test_data = data[training_set_size:]


    # K-Fold split
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
        cls_models = []
        #cls_models.append(GaussianNB())
        cls_models.append(LinearSVC(random_state=0))
        #cls_models.append(RandomForestClassifier(bootstrap=True, max_depth=10, max_features='auto',
        #                                        min_samples_leaf=1, min_samples_split=2, n_estimators=10,
        #                                        random_state=0, n_jobs=-1))
        #cls_models.append(MLPClassifier(activation='tanh', hidden_layer_sizes=(16,)))
        #cls_models.append(KNeighborsClassifier(algorithm='ball_tree', n_neighbors=11, weights='uniform'))
        #cls_models.append(LogisticRegression(class_weight='balanced', fit_intercept=True,
        #solver='liblinear'))

        for clf in cls_models:
            clf.fit(train_data_vectors, train_classes)

        predictions = get_voting(cls_models, dev_data_vectors)
        dev_metrics = get_metrics(predictions, dev_classes)

        predictions = get_voting(cls_models, test_data_vectors)
        test_metrics = get_metrics(predictions, test_classes)

        scores.append(test_metrics)

    print_metrics(scores)

