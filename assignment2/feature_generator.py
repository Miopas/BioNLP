from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

def preprocess_text(raw_text):
    '''
        Preprocessing function
        PROGRAMMING TIP: Always a good idea to have a *master* preprocessing function that reads in a string and returns the
        preprocessed string after applying a series of functions.
    '''
    # stemming and lowercasing (no stopword removal
    #words = [stemmer.stem(w) for w in raw_text.lower().split()]
    #return (" ".join(words))
    return raw_text.lower()


class Record:
    def __init__(self, row, prev_day):
        self.record_id = str(row['record_id'])
        self.age = int(row['age'])
        self.female = 1 if row['female'] == 'Female' else 0
        self.duration = float(row['duration'])
        self.study_day = int(row['fall_study_day'])
        self.location = preprocess_text(row['fall_location'])
        self.text = preprocess_text(row['fall_description'])
        self.label = row['fall_class']
        self.study_interval = self.study_day - prev_day


class FeatureGenerator():
    def __init__(self, train_data):
        train_texts = self.get_texts(train_data)
        train_locations = self.get_locations(train_data)

        # Initialize the tfidf vectorier
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
        self.tfidf_vectorizer.fit(train_texts)

        # Initialize the tfidf vectorier
        self.loc_vectorizer = CountVectorizer(ngram_range=(1, 1))
        self.loc_vectorizer.fit(train_locations)

        return

    def get_texts(self, data):
        res = []
        for x in data:
            res.append(x.text)
        return res

    def get_locations(self, data):
        res = []
        for x in data:
            res.append(x.location)
        return res

    def get_loc_features(self, data):
        locs = self.get_locations(data)
        return self.loc_vectorizer.transform(locs).toarray()

    def get_tfidf_features(self, data):
        texts = self.get_texts(data)
        return self.tfidf_vectorizer.transform(texts).toarray()

    def get_classes(self, data):
        return [r.label for r in data]

    def transform(self, data):
        '''
        - input:
            data: a list of Record objects
        - output:
            vectors: a list of feature vectors
        '''

        all_features = []
        all_features.append(self.get_tfidf_features(data))
        all_features.append(self.get_loc_features(data))
        all_features.append([[r.female] for r in data])
        all_features.append([[r.duration] for r in data])
        all_features.append([[r.study_interval] for r in data])
        all_features = np.concatenate(all_features, axis=1)
        return all_features, self.get_classes(data)
