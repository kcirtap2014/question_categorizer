import pandas as pd
import logging as lg
import numpy as np
from nltk import regexp_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.externals import joblib
import re

import config as CONFIG
import pdb

#from .models import load_joblib, load_data, Airports, Origins, Dests
class LemmaTokenizer(object):
    """
    tokenize text
    """
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stopwords = stopwords
        self.regexp_tokenize = regexp_tokenize

    def __call__(self, doc):
        if pd.notnull(doc):
            # add words to stoplist, previously punctuations have been removed,
            # so we should do the same for the stoplist
            # we also add the top 10 words in the stoplist, these top 10 words
            # are found after post-processing

            doc_lower = self.lower(doc)
            doc_punct = self.striphtmlpunct(doc_lower)
            doc_tabs = self.striptabs(doc_punct)

            # create stoplist
            stoplist = [self.striphtmlpunct(x)
                        for x in self.stopwords.words('english')] + [
                            'im', 'ive'] + [
                            'use', 'get', 'like', 'file', 'would', 'way',
                            'code','work', 'want', 'need']

            lemmatized = []
            regex_tokens = self.regexp_tokenize(doc_tabs,
                                                pattern='\w+\S+|\.\w+')

            for word in regex_tokens:
                #for word, p_tags in pos_tag(regex_tokens):
                #convert_pos_tag = convert_tag(p_tags)
                lemmatized_word = self.wnl.lemmatize(word)
                if lemmatized_word not in set(stoplist):
                    lemmatized.append(lemmatized_word)

            return lemmatized

        return pd.Series(doc)

    def striphtmlpunct(self, data):
        # remove html tags, code unnecessary punctuations
        # <.*?> to remove everything between <>
        # [^\w\s+\.\-\#\+] remove punctuations except .-#+
        # (\.{1,3})(?!\S) negative lookahead assertion: only match .{1,3} that
        # is followed by white space
        if pd.notnull(data):
            p = re.compile(r'<.*?>|[^\w\s+\.\-\#\+]')
            res = p.sub('', data)
            pe = re.compile('(\.{1,3})(?!\S)')

            return pe.sub('', res)
        return data

    def striptabs(self, data):
        # remove tabs breaklines
        p = re.compile(r'(\r\n)+|\r+|\n+|\t+/i')
        return p.sub(' ', data)

    def lower(self, data):
        if pd.notnull(data):
            return data.lower()
        return data

def feature_generator(X, LemmaTokenizer):
    pdb.set_trace()
    vect = joblib.load(CONFIG.DATABASE_URI_VECT)

    X_vect = vect.transform(X)

    return X_vect

def load_model():
    model = joblib.load(CONFIG.DATABASE_URI_MODEL)

    return model

def get_best_tags(y_pred, y_pred_proba, n_tags=1):
    """
    assign at least one tag to y_pred that only have 0

    Parameters:
    -----------
    y_pred: np array
        multilabel predicted y values

    y_pred_proba: np array
        multilabel predicted proba y values

    n_tags: int
        number of non-zero tags

    Returns:
    --------
    y_pred: np array
        new y_pred for evaluation purpose
    """
    y_pred_copy = y_pred.copy()
    idx_y_pred_zeros  = np.where(y_pred_copy.sum(axis=1)==0)[0]
    best_tags = np.argsort(
        y_pred_proba[idx_y_pred_zeros])[:, :-(n_tags + 1):-1]

    for i in range(len(idx_y_pred_zeros)):
        y_pred_copy[idx_y_pred_zeros[i], best_tags[i]] = 1

    return y_pred_copy

def run_predict(title, body):

    X = title + body
    X_vect = feature_generator(X, LemmaTokenizer)
    pdb.set_trace()
    model = load_model()
    y_pred = model.predict(X_vect)
    y_pred_proba  = model.decision_function(X_vect)
    y_pred_new = get_best_tags(y_pred_svm, y_pred_proba, n_tags=2)

    return y_pred_new
