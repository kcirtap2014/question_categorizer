import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib import cm
from nltk import word_tokenize
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import learning_curve, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from collections import defaultdict
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

    if scoring =='mse' or scoring =='rmse':
        score = mean_squared_error
    else:
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

def chi_squared_test(db, series1, series2, alpha=0.05):
    """
    Perform a chi squared test between two categorical series.
    This function is very similar to stats.chi2_contingency()

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

    tab = pd.crosstab(db[series1], db[series2], margins=True)
    observed = tab.iloc[:-1,:-1]   # Get table without totals for later use
    expected = np.outer(tab["All"][:-1], tab.loc["All"][:-1])/db.shape[0]
    expected = pd.DataFrame(expected)
    expected.columns = tab.columns[:-1] # rename columns
    expected.index = tab.index[:-1] # rename index
    chi_squared_stat = (((observed-expected)**2)/expected).sum().sum()
    # sum in row and in columns

    # Find the critical value for 95% confidence*
    crit = stats.chi2.ppf(q=1 - alpha,
                      df = (observed.shape[0] - 1)*(observed.shape[1] - 1))
    # Find the p-value
    p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,
                             df=(observed.shape[0] - 1)*(observed.shape[1] - 1))
    if p_value<alpha:
        concl = ("p_value: %.2E<0.05 (alpha), we reject the null hypothesis."
                 %p_value)
    else:
        concl = ("p_value: %.2E>0.05 (alpha), we accept the null hypothesis."
                 %p_value)

    return (chi_squared_stat, crit, p_value, concl )
