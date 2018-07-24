import pandas as pd
import numpy as np
import pickle
from lemma_tokenizer import LemmaTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import NMF
import networkx as nx

from helper_functions import (evaluate, get_best_tags, potential_tags,
                              topic_name_attribution)
import pdb

class Model(object):
    """
    this class consists of modules that are needed to train and evaluate a model.
    The model can be of supervised learning or unsupervised learning.

    """

    def __init__(self, model, vectorizer, binarizer, G_tags):
        self.model = model
        self.vectorizer = vectorizer
        self.binarizer = binarizer
        self.G_tags = G_tags

    def create_pipeline(self):
        """
        create pipeline
        """
        self.pipeline = Pipeline([('vect', self.vectorizer),
                                  ('clf', self.model)])

    def fit(self, X, y):
        """
        train the model

        Parameters:
        -----------
        X: DataFrame
            input of training set

        y: DataFrame
            output of training set
        """
        self.pipeline.fit(X, y)

    def predict(self, X):
        """
        predict the y for X

        Parameters:
        -----------
        X: DataFrame
            input for prediction

        Returns:
        --------
        y: sparse matrix (2D)
            predicted values
        """
        y_pred = self.pipeline.predict(X)
        y_pred_proba = self.pipeline.predict_proba(X)
        y_pred_new = get_best_tags(y_pred, y_pred_proba)

        return y_pred_new

    def f1_score(self, y_true, y_pred):
        """
        evaluate the f1_score of the model, print the score,
        act as a sanity check

        Parameters:
        -----------
        y_true: 2d arrays
            true labels
        y_pred: 2d arrays
            predicted labels
        """
        score_val = evaluate(y_true, y_pred,
                             self.binarizer, self.G_tags,
                             l_print_errors=False, l_deduplication = True)

        print('Test score: {0:.2f}'.format(score_val))

    def save(self, filename_vect, filename_clf):
        """
        save both vectorizer and classifier for deployment

        Parameters:
        -----------
        filename_vect: str
            complete filename for vectorizer

        filename_clf: str
            complete filename for model
        """
        pickle.dump(self.pipeline.named_steps["vect"], open(filename_vect, 'wb'))
        print("Saved vectorizer to %s" %filename_vect)
        pickle.dump(self.pipeline.named_steps["clf"], open(filename_clf, 'wb'))
        print("Saved model to %s" %filename_clf)

    def save_topics(self, filename_topics):
        """
        save topics

        Parameters:
        -----------
        filename_topics: str
            complete filename for topics

        """
        pickle.dump((self.dict_topicnames, self.topicnames),
                    open(filename_topics, 'wb'))
        print("Saved model to %s" %filename_topics)

    def attribute_topic_names(self):
        """
        attribute topics names
        """
        feat_names = self.pipeline.named_steps["vect"].get_feature_names()
        df_top_words = pd.DataFrame(
                                self.pipeline.named_steps["clf"].components_,
                                columns=feat_names)
        tags_keys = self.binarizer.classes_
        self.dict_topicnames, self.topicnames = topic_name_attribution(
                                                df_top_words, tags_keys)

    def attribute_topic_documents(self, X):
        """
        attribute topics to documents
        Parameters:
        -----------
        X: DataFrame
            input

        Returns:
        --------
        dominant_topics: arrays
            dominant_topics for each entry
        """
        output = self.pipeline.transform(X)
        df_document_topic = pd.DataFrame(np.round(output, 2),
                                         columns=self.topicnames,
                                         index=X.index)
        dominant_topic = df_document_topic.apply(potential_tags, axis=1)

        return dominant_topic

if __name__ == '__main__':
    print("Loading files:")

    X_train = pd.read_csv("X_train.csv", index_col = "Id")
    X_train_all = pd.read_csv("X_train_nmfkl.csv", index_col = "Id")
    X_train_nmfkl = X_train_all[X_train_all['0'].notnull()]
    X_test = pd.read_csv("X_test.csv", index_col = "Id")
    index_y_all = np.load("index_y_true_nmf.npy")
    y_all = np.load("y_true_nmf.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")

    # load binarizer
    lb = joblib.load("binarizer.pk")
    mlb = joblib.load("new_binarizer.pk")

    # file directory
    file_dir = "/Users/pmlee/Documents/CAPGemini_OpenClassroom/" + \
               "OpenClassrooms_Patrick_Lee/Assignment5/question_categorizer/" + \
               "tags_recommender_app/TagsRecommenderApp/static/db/"

    # load networkx graph
    G_tags = nx.read_gpickle("G_tags.gpickle")

    print("Training SVM model")
    params_vectorizer_tfidf = {
                    "max_features": 5000,
                    "ngram_range": (1, 2),
                    'tokenizer': LemmaTokenizer(),
                    'lowercase': False
                    }
    params_model_svm = {"kernel": "linear", "C": 0.01}
    tfidf_vectorizer = TfidfVectorizer(**params_vectorizer_tfidf)
    svm_model = OneVsRestClassifier(CalibratedClassifierCV(SVC(**params_model_svm)))
    svm_clf = Model(svm_model, tfidf_vectorizer, lb, G_tags)

    print("Fitting the SVM model tf-idf features...")
    svm_clf.create_pipeline()
    svm_clf.fit(X_train['0'], y_train)

    # prediction
    y_pred_svm = svm_clf.predict(X_test['0'])
    svm_clf.f1_score(y_test, y_pred_svm)

    # save file
    filename_vect = file_dir + "vectorizer_svm.pk"
    filename_clf = file_dir + "OVR_SVM_model.sav"
    pdb.set_trace()
    svm_clf.save(filename_vect, filename_clf)

    print("Training PLSA model")

    n_chosen_components = 5200
    n_top_words = 20
    n_topics = 300

    params_vectorizer_count = {
        "max_features": n_chosen_components,
        "ngram_range": (1, 1),
        'tokenizer': LemmaTokenizer(),
        'lowercase': False
    }

    params_nmf = {
        "n_components": n_topics,
        "beta_loss": "kullback-leibler",
        "solver": 'mu',
        "max_iter": 1000,
        "alpha": .1,
        "l1_ratio": .5
    }

    count_vectorizer = CountVectorizer(**params_vectorizer_count)
    clf_nmf = NMF(**params_nmf)
    nmf_clf = Model(clf_nmf, count_vectorizer, lb, G_tags)

    print("Fitting the NMF model (KL divergence) with "
      "count features, num_topics =%d..." % n_topics)
    nmf_clf.create_pipeline()
    nmf_clf.fit(X_train_nmfkl['0'], None)
    nmf_clf.attribute_topic_names()
    dominant_topic = nmf_clf.attribute_topic_documents(X_train_nmfkl['0'])
    y_pred_nmf = mlb.fit_transform(dominant_topic.loc[index_y_all])

    # evaluate model
    nmf_clf.f1_score(y_all, y_pred_nmf)
    y_pred_nmf_comp = mlb.fit_transform(dominant_topic)
    no_tag_score_nmfkl = no_tag_percentage_score(y_pred_nmf_comp, mlb)
    print('No tag score: {0:.2f}'.format(no_tag_score_nmfkl))

    filename_vect = file_dir + "vectorizer_plsa.pk"
    filename_clf = file_dir + "PLSA_model.sav"
    filename_topics = file_dir + "topicnames.pk"
    pdb.set_trace()
    nmf_clf.save(filename_vect, filename_clf)
    nmf_clf.save_topics(filename_topics)
