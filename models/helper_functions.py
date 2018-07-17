import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib import cm
from nltk import word_tokenize, FreqDist
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import learning_curve, KFold, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import jaccard_similarity_score, f1_score
import math
from sklearn.pipeline import Pipeline
from collections import defaultdict, Counter
from operator import itemgetter
from scipy.stats import entropy
from sklearn.calibration import CalibratedClassifierCV

import pdb

class ModelTrainer(object):
    """
    a class that trains models, plot learning curve, and returns scores
    """

    def __init__(self,
                 model,
                 X_train,
                 X_test,
                 train_sizes,
                 title,
                 kfold = 10,
                 scoring = 'rmse',
                 params = None):
        self.model = model
        self.kfold = kfold
        self.train_sizes = train_sizes
        self.n_cut = len(train_sizes)
        self.scoring = scoring
        self.params = params
        self.title = title

    def grid_search(self):
        g_model = GridSearchCV(self.model, self.params,
                               cv=self.kfold, return_train_score=True)
        self.model = g_model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def learning_curve(self, X_train, X_test, y_train, y_test):
        fig, ax = plt.subplots()

        ax = plot_learning_curve(self.model, self.title,
                                 X_train, y_train, X_test, y_test,
                                 n_cut=self.n_cut,
                                 train_sizes=self.train_sizes,
                                 scoring=self.scoring, ax=ax)
        return fig, ax

    def predict(self, X_train, X_test):
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        return y_pred_train, y_pred_test

    def evaluate(self, y_train, y_test, y_pred_train, y_pred_test):
        rmse_score_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_score_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        return rmse_score_train, rmse_score_test

    def run(self, X_train, X_test, y_train, y_test):
        if self.params is not None:
            self.grid_search()

        self.fit(X_train, y_train)
        fig, ax = self.learning_curve(X_train, X_test, y_train, y_test)
        y_pred_train, y_pred_test = self.predict(X_train, X_test)
        rmse_score_train, rmse_score_test = self.evaluate(y_train,
                                                          y_test,
                                                          y_pred_train,
                                                          y_pred_test)

        return fig, ax, y_pred_train, y_pred_test, rmse_score_train, rmse_score_test

class CrossValidation(object):
    def __init__(self,
                 classifier,
                 vectorizer,
                 params,
                 binarizer,
                 G_tags,
                 n_splits=3,
                 n_tags=1,
                 score_threshold=0.5,
                 l_deduplication=False,
                 l_print_errors=False):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.params = params
        self.n_splits = n_splits
        self.binarizer = binarizer
        self.G_tags = G_tags
        self.n_tags = n_tags
        self.score_threshold = score_threshold
        self.l_print_errors = l_print_errors
        self.l_deduplication = l_deduplication

    def _construct_pipeline(self):

        pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('clf', self.classifier),])
        self.pipeline = pipeline

    def cv(self, X_train, y_train):
        kf = KFold(n_splits=self.n_splits)

        key = list(self.params.keys())[0]
        cv_scores = [[] for i in range(len(self.params[key]))]
        mean_cv_dict = dict()
        self._construct_pipeline()

        for idx, value in enumerate(self.params[key]):

            params = {key: value}

            for train_index, test_index in kf.split(X_train, y_train):

                X_train_train = X_train.iloc[train_index]
                y_train_train = y_train[train_index,:]

                X_train_test = X_train.iloc[test_index]
                y_train_test = y_train[test_index,:]

                self.pipeline.named_steps['clf'].estimator.set_params(**params)
                self.pipeline.fit(X_train_train, y_train_train)

                y_pred = self.pipeline.predict(X_train_test)
                y_pred_proba = self.pipeline.predict_proba(X_train_test)
                y_pred_new = get_best_tags(y_pred, y_pred_proba, n_tags = self.n_tags)
                cv_scores[idx].append(evaluate(y_train_test, y_pred_new,
                                               binarizer=self.binarizer,
                                               G_tags=self.G_tags,
                                               score_threshold=self.score_threshold,
                                               l_print_errors=self.l_print_errors,
                                               l_deduplication=self.l_deduplication))

            mean_cv_dict[value] = np.mean(cv_scores[idx])

        sorted_mean_cv_dict = sorted(
            mean_cv_dict.items(), key=itemgetter(1), reverse=True)
        self.best_parameter_ = {key: {"value": sorted_mean_cv_dict[0][0],
                                      "mean": sorted_mean_cv_dict[0][1]}}

class CountEmbeddingVectorizer(object):
    """
    vectorizer that incorporate some nltk features: word2vec
    """

    def __init__(self,
                 word2vec,
                 stop_words='english',
                 ngram_range=(1, 1),
                 max_df=None,
                 min_df=None):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(w2v.values())))
        self.max_df = max_df
        self.min_df = min_df
        self.stop_words = stop_words
        self.ngram_range = ngram_range

    def fit(self, X):

        vectorizer = CountVectorizer(
            max_df=self.max_df,
            min_df=self.min_df,
            stop_words='english',
            ngram_range=self.ngram_range)
        vectorizer.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        self.word2weight = vectorizer.vocabulary_

        return self

    def transform(self, X):

        tf = []

        for words in X:

            try:
                tf.append(
                    np.mean(
                        [
                            self.word2vec[w] * self.word2weight[w]
                            for w in word_tokenize(words)
                        ],
                        axis=0))
                #pdb.set_trace()
            except KeyError:
                tf.append(np.zeros(self.dim))

        tf = np.array(tf)

        return tf

def checkPairFeatures(df, featureA, featureB):
    """
    Check if at least one of the entry features has a non null value.

    Parameters:
    -----------
    df: pandas dataframe
        input dataframe

    featureA : str
        name of feature A

    featureB : str
        name of feature B

    Return:
    -------
    df: pandas dataframe
        filtered database
    """
    db = df[~np.logical_and(df[featureA].isnull(), df[featureB].isnull())]

    return db

def unique_values(df):
    """
    return unique values of each column in dataset df

    Parameters:
    -----------
    df: pandas dataframe
        input dataframe

    Return:
    ------
    count: dict
        column and corresponding count of unique values
    """
    count = {}

    for i in df.columns:
        count[i] = df[i].nunique()

    return count

def group_sort(df, feature, by_index=False, ascending=True, lim=None):
    """
    Group by features, and sort that in descending order with respect to size

    Parameters:
    -----------
    df: pandas dataframe
        input dataframe

    feature : str
        name of feature

    ascending: boolean
        ascending or descending order

    lim: int
        Default: None

    Return:
    ------
    db_sorted: pandas dataframe
        filtered_database
    """
    db = df.groupby([feature]).size()

    if by_index:
        db_sorted = db.sort_index(ascending=ascending)

    else:
        db_sorted = db.sort_values(ascending=ascending)

    if lim is not None:
        db_sorted = db_sorted.head(lim)

    return db_sorted


def check_crosstab(db, series1, series2, lim = 5, normalize = False):
    """
    perform pd.crosstab and return a crosstab with every column
    count>lim.

    Parameters:
    -----------
    db: pandas DataFrame
        input DataFrame

    series1: str
        name of the categorical variable

    series1: str
        name of the other categorical variable

    Returns:
    --------
    tab_m5: crosstab
        crosstab with every column count>lim, without margins
    """

    tab = pd.crosstab(db[series1], db[series2], margins=True)

    series1_m5 = tab.iloc[:,-1][tab.iloc[:,-1]>=lim].index
    series2_m5 = tab.iloc[-1,:][tab.iloc[-1,:]>=lim].index

    if normalize == "index":
        norm_index = tab.loc[ "All", :]
        lim_index = lim/norm_index
        tab = tab/np.tile(norm_index, (tab.shape[0],1))
        series2_m5 = tab.iloc[-1,:][tab.iloc[-1,:]>=lim_index].index

    elif normalize == "columns":
        norm_column = tab.loc[ :, "All"]
        lim_column = lim/norm_column
        tab = tab/np.tile(norm_column, (tab.shape[1],1)).T
        series1_m5 = tab.iloc[:,-1][tab.iloc[:,-1]>=lim_column].index

    elif normalize == "all":
        norm = tab.loc["All", "All"]
        tab = tab/norm
        series2_m5 = tab.iloc[-1,:][tab.iloc[-1,:]>=lim].index
        series1_m5 = tab.iloc[:,-1][tab.iloc[:,-1]>=lim].index

    tab_m5 = tab.loc[series1_m5, series2_m5]

    return tab_m5

def cat_corr(db, series1, series2, alpha=0.05, method="chi2",
            lim=5, normalize=False):
    """
    Perform a correlation measure between two categorical series.

    Parameters:
    -----------
    db: pandas DataFrame
        input DataFrame

    series1: str
        name of the categorical variable

    series1: str
        name of the other categorical variable

    alpha: float

    Returns:
    --------
    chi_squared_stat: float
        computed chi squared value from the dataset

    crit: float
        critical value to indicate the critical region

    p_value: float
        p_value of the test

    conclusion: str
        conclusion of the test
    """

    tab = check_crosstab( db, series1, series2, lim=lim, normalize=normalize )
    N = tab.iloc[:-1, :-1].sum().sum() # sum in row and in column

    if method == "chi2":
        # tab[:-1, :-1] to not take the margin
        chi2, p_value, ddf, expt = stats.chi2_contingency(tab.iloc[:-1, :-1])

        # Find the critical value for 95% confidence*
        crit = stats.chi2.ppf(q=1 - alpha, df=ddf)

        return chi2, crit

    elif method == "cramerV":

        chi2, p_value, ddf, expt = stats.chi2_contingency(tab.iloc[:-1,:-1])
        print (chi2, N*(np.min(tab.iloc[:-1, :-1].shape)-1))
        V = np.sqrt(chi2/(N*(np.min(tab.iloc[:-1, :-1].shape)-1)))

        return V

    elif method == "kruskalT":

        # marginals
        pi_s1 = tab.iloc[:-1,:-1].sum(axis=1)/N
        pi_s2 = tab.iloc[:-1,:-1].sum(axis=0)/N
        pi_s1_s2 = np.outer(pi_s1, pi_s2)
        pi_n = tab.iloc[:-1,:-1]/N

        # Kruskal's T coefficient
        num_tau = ((pi_n**2 - pi_s1_s2**2)/pi_s2).sum().sum()
        denom_tau = 1 - (pi_s2**2).sum()
        tau = num_tau/denom_tau

        return tau

    elif method == "theilU":

        # H_s2 marginal entropy
        S_s1 = tab.iloc[:-1,:-1].sum(axis=1)/N
        S_s1_log = np.log(S_s1)

        # impose log(0) = 0
        S_s1_log_masked = np.ma.masked_invalid(S_s1_log)
        S_s1_log_masked.filled(0.0)
        H_s2 = -(S_s1*S_s1_log_masked).sum(axis=0)

        # H_s2_s1 conditional entropy
        S_s2 = tab.iloc[:-1,:-1].sum(axis=0)
        S_c_lin = tab.iloc[:-1,:-1]/S_s2
        S_c_log = np.log(S_c_lin)
        S_c_log_masked = np.ma.masked_invalid(S_c_log)
        S_c_log_masked.filled(0.0)
        S_c = (S_c_lin*S_c_log_masked).sum(axis = 0)
        H_s2_s1 = - (S_s2*S_c).sum()/N

        # Theil's coefficient
        U = (H_s2 - H_s2_s1)/H_s2

        return U

def graph_silhouette(sample_silhouette_values, cluster_labels, n_clusters,
                    silhouette_avg, ax):
    """
    draw the silhouette graph plots

    Parameters:
    -----------
    sample_silhouette_values: np.array
        sample_silhouette_values obtained from
        sklearn.metrics.silhouette_samples

    cluster_labels: np.array
        cluster label for each sample point

    n_clusters: int
        number of clusters

    silhoeutte_avg: float
        average silhouette value

    Return:
    -------
    ax: matplotlib axis
        axis of the plot
    """
    y_lower = 10

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples


    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    return ax

def one_hot_encoder(df, feature_name, prefix):
    """
    Generate one hot-encoding of certain data and concatenate back to
    the input df

    Parameters:
    -----------
    df : pandas DataFrame
        input DataFrame

    feature_name : string
        feature that is to be encoded

    prefix : string
        prefix of the new feature generated due to one_hot_encoding

    Returns:
    --------
    df : pandas DataFrame
    """
    df_dummy = pd.get_dummies(
        df[feature_name], prefix=prefix)
    df = df.drop(
        feature_name, axis=1).copy()

    for dummy in df_dummy.columns:
        df.loc[:, dummy] = df_dummy[dummy]

    return df

def plot_learning_curve(estimator, title, X, y, X_test, y_test, ylim=None,
                        n_cut=5, n_splits=5,
                        train_sizes=np.linspace(.1, 1.0, 5),
                        scoring='r2', ax=None):
    """
    Generate a simple plot of the test and training learning curve. Adapted
    from http://scikit-learn.org/stable/auto_examples/model_selection
    /plot_learning_curve.html

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    train_sizes: array_like
        determines the total number of training
    """
    if ax is None:
        fig, ax=  plt.subplots(1, figsize=(3,2.5))

    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")

    train_scores = [[] for i in range(n_cut)]
    cv_scores = [[] for i in range(n_cut)]
    test_scores = [[] for i in range(n_cut)]
    sample_sizes = []

    if callable(scoring):
        score = scoring
    else:
        if scoring =='mse' or scoring =='rmse':
            score = mean_squared_error
        elif scoring == 'r2':
            score = r2_score

    #test_size = len(X_test)/len(X)

    for i, size in enumerate(train_sizes):
        sample_sizes.append(int(size*len(X)))
        X_temp = X.sample(sample_sizes[i], random_state = 3)
        #X_test_temp = X_test.sample(int(test_size*sample_sizes[i]),
        #                            random_state = 43)
        y_temp = y[X_temp.index]
        #y_test_temp = y_test[X_test_temp.index]
        kf = KFold(n_splits=n_splits)

        for train_index, test_index in kf.split(X_temp):
            estimator.fit(X_temp.iloc[train_index], y_temp.iloc[train_index])

            # training
            y_train_pred = estimator.predict(X_temp.iloc[train_index])
            train_scores[i].append(score(y_temp.iloc[train_index], y_train_pred))

            # cross validation
            y_cv_pred = estimator.predict(X_temp.iloc[test_index])
            cv_scores[i].append(score(y_temp.iloc[test_index], y_cv_pred))

            # test
            X_test_temp = X_test.sample(len(X_temp.iloc[test_index]),
                            random_state=3)
            y_test_temp = y_test[X_test_temp.index]
            y_test_pred = estimator.predict(X_test_temp)
            test_scores[i].append(score(y_test_temp, y_test_pred))

    #train_sizes, train_scores, cv_scores = learning_curve(
    #    estimator, X, y, cv=cv, n_jobs=n_jobs, scoring=scoring,
    #    train_sizes=train_sizes)

    # predictor on the test set
    #test_sizes, test_score,  learning_curve(
    #    estimator, X_test, y_test, n_jobs=n_jobs, scoring=scoring,
    #    train_sizes = 1)

    if scoring == 'rmse':
        train_scores = np.sqrt(train_scores)
        cv_scores = np.sqrt(cv_scores)
        test_scores = np.sqrt(test_scores)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    cv_scores_std = np.std(cv_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.grid()

    ax.fill_between(sample_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax.fill_between(sample_sizes, cv_scores_mean - cv_scores_std,
                     cv_scores_mean + cv_scores_std, alpha=0.1, color="g")
    ax.fill_between(sample_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="b")
    ax.plot(sample_sizes, train_scores_mean, 'o-', color="r",
             label="Training")
    ax.plot(sample_sizes, cv_scores_mean, 'o-', color="g",
             label="Cross-validation")
    ax.plot(sample_sizes, test_scores_mean, 'o-', color="b",
             label="Test")

    ax.legend(loc='best')

    return ax

def freq_stats_corpora(df):
    """
    return freq and stats of a corpora

    Parameters:
    -----------
    df: pandas dataframe
        input dataframe

    Returns:
    --------
    freq: dict
        FreqDist of the word

    stats: dict
        number of total and unique word distributions

    corpora: defaultdict
        corpus of the word

    """
    corpora = defaultdict(list)

    for tags_id, el in df.iterrows():
        for tag in el.elkey:
            corpora[tag] += el.elvalue

    stats, freq = dict(), dict()

    for k,v in corpora.items():
        freq[k] = FreqDist(v)
        stats[k] = {'total':len(v), 'unique':len(freq[k].keys())}

    return (freq, stats, corpora)

def ngram(a, n):
    """
    returns an ngram of a word

    Parameters:
    -----------
    a: string
        a word

    n: int
        ngram 2 for bigram, 3 for trigram

    Returns:
    --------
    a_list: array
        list of ngrams of the word
    """

    a_list = []
    j = n

    for i in range(0,len(a)-(n-1)):
        a_list.append(a[i:j])
        j+=1

    return a_list

def jaccard_distance(a,b):
    """
    returns jaccard distance between 2 arrays of ngrams

    Parameters:
    -----------
    seta: array
        ngrams of word a
    setb: array
        ngrams of word b

    Returns:
    --------
    score_jacc: float
        jaccard distance of both words
    """

    union = set(a).union(set(b))
    inter = set(a).intersection(set(b))
    len_union = len(union)

    if len(union) == 0:
        len_union = 1
    score_jacc = len(inter)/len_union

    return score_jacc

def cosine_similarity(a, b):
    """
    returns cosine similarity of ngrams of word a and ngrams of word b

    Parameters:
    -----------
    a: array
        ngrams of word a

    b: array
        ngrams of word b

    Returns:
    --------
    cosine_sim: float
        cosine similarity of both words
    """

    intersection = set(a) & set(b)
    veca = Counter(a)
    vecb = Counter(b)
    numerator = sum([veca[x] * vecb[x] for x in intersection])

    sum1 = sum([veca[x]**2 for x in veca.keys()])
    sum2 = sum([vecb[x]**2 for x in vecb.keys()])

    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def check_similar_tags(a, b, G_tags):
    """
    check if tag a belongs to the same group as tag b

    Parameters:
    -----------
    a: string
        tag a

    b: string
        tag b

    sim_tags: networkx graph
        graph of tags

    Returns:
    --------
    boolean
        true if tag a and tag b belong to the same group
    """

    a_edge_list = [i[1] for i in G_tags.edges(a)]
    b_edge_list = [i[1] for i in G_tags.edges(b)]

    if (b in a_edge_list) and (a in b_edge_list):
        return True
    else:
        return False


def evaluate(y_true,
             y_pred,
             binarizer=None,
             G_tags=None,
             score_threshold=0.5,
             l_print_errors=False,
             l_deduplication=False):
    """
    returns accuracy score. In the case of crossval, returns errors in cross
    evaluation

    Parameters:
    -----------
    X: numpy array, or scipy sparse array
        entry features

    y: numpy array,
        multilabel y true labels

    classifier: scikit learn model
        trained model

    binarizer: multi label binarizer object
        to inverse binarized object to feature names

    G_tags: networkx graph
        used in tags regrouping

    score_threshold: float
        score threshold to decide if to consider as an error

    l_print_errors: boolean
        True if print errors

    l_deduplication: boolean
        remove duplicated tags in true and pred labels

    Returns:
    --------
    accuracy_score: float

    errors: array
        errors
    """

    f1 = 0.
    errors = []

    if binarizer is None:
        raise "Binarizer is None"

    if G_tags is None:
        raise "G_tags is None"

    for index in range(len(y_true)):

        y_pred_temp = y_pred[index].reshape(1, -1)
        y_true_temp = y_true[index].reshape(1, -1)

        y_true_tag = binarizer.inverse_transform(y_true_temp)
        y_pred_tag = binarizer.inverse_transform(y_pred_temp)

        index_true = np.where(y_true_temp.flatten() == 1)[0]
        index_pred = np.where(y_pred_temp.flatten() == 1)[0]
        union_index = list(set(index_true).union(set(index_pred)))

        if l_deduplication:
            # check for duplicated true values
            for p in range(len(y_true_tag[0])):
                for q in range(p + 1, len(y_true_tag[0])):
                    if check_similar_tags(y_true_tag[0][p], y_true_tag[0][q],
                                          G_tags):
                        y_true_temp[0, index_true[q]] = 0

            # check for duplicated predictions
            for p in range(len(y_pred_tag[0])):
                for q in range(p + 1, len(y_pred_tag[0])):
                    if check_similar_tags(y_pred_tag[0][p], y_pred_tag[0][q],
                                          G_tags):
                        y_pred_temp[0, index_pred[q]] = 0

        y_true_tag_dedup = binarizer.inverse_transform(y_true_temp)
        y_pred_tag_dedup = binarizer.inverse_transform(y_pred_temp)

        for i in range(len(y_true_tag_dedup)):
            for j in range(len(y_pred_tag_dedup)):
                if check_similar_tags(y_true_tag_dedup[i], y_pred_tag_dedup[j],
                                      G_tags):
                    y_true_temp[0, union_index[i]] = 1
                    y_pred_temp[0, union_index[j]] = 1

        #y_true_eval = y_true_temp[0, union_index]
        #y_pred_eval = y_pred_temp[0, union_index]

        score_f1 = f1_score(y_true_temp.flatten(), y_pred_temp.flatten())
        f1 += score_f1
        #jac_sim = jaccard_similarity_score(
        #    y_true_eval, y_pred_eval, normalize=False)
        # if only half of the total are mislabeled,
        # then we consider that it's correct
        if l_print_errors:
            if score_f1 < score_threshold:
                errors.append((binarizer.inverse_transform(y_true_temp),
                               binarizer.inverse_transform(y_pred_temp)))

        #if jac_sim >= n_correct:
        #    accuracy += 1.

    if l_print_errors:
        return f1 / y_true.shape[0], errors
    else:
        # normalize accuracy to the number of samples
        return f1 / y_true.shape[0]

def get_best_tags(y_pred, y_pred_proba, n_tags=2):
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

def deduplication(y_pred_tags, binarizer, G_tags):
    """
    returns deduplicated y_pred

    Parameters:
    -----------
    y_pred_tags: np array
        y arrays that are to be deduplicated, normally y_pred

    binarizer: multilabel binarizer object
        use for inverse transform y_pred to tags

    G_tags: network graph
        graph for similar tags identification

    Returns:
    --------
    similar_tags_index: dict
    """

    similar_tags_index = dict()

    for i in range(len(y_pred_tags)):
        temp_array = []
        for j in range(i + 1, len(y_pred_tags)):
            if y_pred_tags[i] in [i[1] for i in G_tags.edges(y_pred_tags[j])]:
                temp_array.append(idx_y_pred[i])
        similar_tags_index[y_pred_tags[i]] = temp_array

    return similar_tags_index

def H_entropy(x):
    """
    returns the information entropy of x, useful for topic pruning
    high entropy values represent high uncertainties, hence the lower the better

    Parameters:
    -----------
    x: np arrays

    Returns:
    --------
    entropy: float
    """

    return -sum(x*np.log(np.ma.array(x)))


def rpc_score(perplexity, topic_num):
    """
    returns rate of perplexity change

    Parameters:
    -----------
    perplexity: np arrays
        perplexity given by LDA

    topic_num: np arrays
        array of topic num

    Returns:
    --------
    rpc: np arrays
        rate of perplexity change
    """

    return np.absolute((perplexity[1:] - perplexity[:-1]) / (topic_num[1:] - topic_num[:-1]))

def print_top_words(model, feature_names, n_top_words):
    """
    print top words

    Parameters:
    -----------
    model: sciktlearn decomposition model

    feature_names: output of tfidf or count vectorizer feature names

    n_top_words: int
        number of words to be printed
    """

    for topic_idx, topic in enumerate(model.components_):
        sorted_data = sorted(
            list(zip(topic, feature_names)),
            key=itemgetter(0),
            reverse=True)
        message = "Topic #%d: " % topic_idx
        message += " ".join([i[1] for i in sorted_data[:n_top_words]])
        print(message)
    print()

def topic_name_attribution(df_top_words, tags_key):
    """
    attribute topic names to each topic
    """
    words = [[] for i in range(df_top_words.shape[0])]
    Hs = dict()
    n_top = 100
    num_keep_words = 10

    for k in df_top_words.index:
        df_top_words_sorted = df_top_words.loc[k].sort_values(ascending=False)
        Hs[k] = entropy(df_top_words_sorted.values)

        for word in df_top_words_sorted[:n_top].index:

            if word in set(tags_key):
                words[k].append(word)

        # default values
        if not words[k]:
            words[k] = ['unknown']
            #df_top_words_sorted[:num_keep_words].index.tolist()

    index_Hs_neg = [k for k, v in Hs.items() if (v < np.max(list(Hs.values())) and v>0.)]
    dict_topicnames = defaultdict(list)
    topicnames = []

    # arrays for significant topics only
    for idx, word in enumerate(np.array(words)[index_Hs_neg]):
        if len(word)>2:
            word = word[:2]

        groupword = ','.join(word)
        dict_topicnames[index_Hs_neg[idx]].append(groupword)
        topicnames.append(groupword)

    return dict_topicnames, topicnames

def no_tag_percentage_score(y, binarizer):
    """
    return no_tag_percentage_score

    Parameters:
    -----------
    y: np array
        input

    binarize: multilabel binarizer

    Returns:
    --------
    float
        no tag percentage score
    """
    count=0

    for index in range(len(y)):
        y_temp = y[index].reshape(1, -1)
        y_tags = binarizer.inverse_transform(y_temp)

        if y_tags[0][0]=="No tags":
            count+=1

    return float(count)/len(y)
