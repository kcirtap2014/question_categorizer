import pandas as pd
import logging as lg
import numpy as np
import config as CONFIG
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from lemma_tokenizer import LemmaTokenizer

def feature_generator(df_X_input):
    lt = LemmaTokenizer()
    vect = joblib.load(CONFIG.DATABASE_URI + "vectorizer_lemma2.pk")
    X_trans = vect.transform(df_X_input)

    return X_trans

def load_model():
    model = joblib.load(CONFIG.DATABASE_URI+ "OVR_SVM_model.sav")

    return model

def get_best_tags(y_pred, y_pred_proba, n_tags=3):
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
    idx_y_pred_zeros  = np.where(y_pred_copy.sum(axis=1)<n_tags)[0]
    best_tags = np.argsort(
        y_pred_proba[idx_y_pred_zeros])[:, :-(n_tags + 1):-1]

    for i in range(len(idx_y_pred_zeros)):
        y_pred_copy[idx_y_pred_zeros[i], best_tags[i]] = 1

    return y_pred_copy

def load_binarizer():
    binarizer = joblib.load(CONFIG.DATABASE_URI + "binarizer.pk")

    return binarizer

def run_predict(title, body):

    # prepare X input
    X = str(title) + " " + str(body)
    df_X = pd.Series([X])
    X_trans = feature_generator(df_X)

    # load model for prediction
    model = load_model()
    y_pred = model.predict(X_trans)
    y_pred_proba  = model.predict_proba(X_trans)
    y_pred_new = get_best_tags(y_pred, y_pred_proba)

    # load binarizer to convert prediction to tags
    binarizer = load_binarizer()
    rec_tags = binarizer.inverse_transform(y_pred_new)

    return rec_tags
