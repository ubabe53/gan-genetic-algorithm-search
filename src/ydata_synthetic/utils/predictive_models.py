"""Predictive model.
Reference: J. Jordon, J. Yoon, M. van der Schaar,
           "Measuring the quality of Synthetic data for use in competitions,"
           KDD Workshop on Machine Learning for Medicine and Healthcare, 2018
Paper Link: https://arxiv.org/abs/1806.11345
Contact: jsyoon0823@gmail.com
"""

# Imports
import numpy as np

from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, Ridge, LinearRegression, Lasso
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error, \
    f1_score, accuracy_score, r2_score, median_absolute_error

def predictive_model_classification(train_x, train_y, test_x, model_name):
    """Predictive model define, train, and test.

    Args:
      - train_x: training features
      - train_y: training labels
      - test_x: testing features
      - model_name: predictive model name

    Returns:
      - test_y_hat: prediction on testing set
    """

    assert model_name in ['logisticregression', 'nn', 'randomforest',
                          'gaussiannb', 'bernoullinb', 'multinb',
                          'svmlin', 'gbm', 'extra trees',
                          'lda', 'passive aggressive', 'adaboost',
                          'bagging', 'xgb'], 'Please put an existing model'

    # Define model
    if model_name == 'logisticregression':
        model = LogisticRegression()
    elif model_name == 'nn':
        model = MLPClassifier(hidden_layer_sizes=(200, 200))
    elif model_name == 'randomforest':
        model = RandomForestClassifier()
    elif model_name == 'gaussiannb':
        model = GaussianNB()
    elif model_name == 'bernoullinb':
        model = BernoulliNB()
    elif model_name == 'multinb':
        model = MultinomialNB()
    elif model_name == 'svmlin':
        model = svm.LinearSVC()
    elif model_name == 'gbm':
        model = GradientBoostingClassifier()
    elif model_name == 'extra trees':
        model = ExtraTreesClassifier(n_estimators=20)
    elif model_name == 'lda':
        model = LinearDiscriminantAnalysis()
    elif model_name == 'passive aggressive':
        model = PassiveAggressiveClassifier()
    elif model_name == 'adaboost':
        model = AdaBoostClassifier()
    elif model_name == 'bagging':
        model = BaggingClassifier()


        # Train & Predict
    if model_name in ['svmlin', 'Passive Aggressive']:
        model.fit(train_x, train_y)
        test_y_hat = model.predict(test_x)
    else:
        model.fit(train_x, train_y)
        test_y_hat = model.predict(test_x)

    return test_y_hat


def performance(test_y, test_y_hat, metric_name):
    """Evaluate predictive model performance.

    Args:
      - test_y: original testing labels
      - test_y_hat: prediction on testing data
      - metric_name: 'auc' or 'apr'

    Returns:
      - score: performance of the predictive model
    """

    assert metric_name in ['auc', 'apr', 'mae', 'rsme', 'f1', 'accuracy', 'r2', 'meae'], 'Please put an existing metric'

    if metric_name == 'auc':
        score = roc_auc_score(test_y, test_y_hat)
    elif metric_name == 'apr':
        score = average_precision_score(test_y, test_y_hat)
    elif metric_name == 'mae':
        score = mean_absolute_error(test_y, test_y_hat)
    elif metric_name == 'rsme':
        score = mean_squared_error(test_y, test_y_hat, squared=False)
    elif metric_name == 'f1':
        score = f1_score(test_y, test_y_hat)
    elif metric_name == 'accuracy':
        score = accuracy_score(test_y, test_y_hat)
    elif metric_name == 'r2':
        score = r2_score(test_y, test_y_hat)
    elif metric_name == 'meae':
        score = median_absolute_error(test_y, test_y_hat)

    return score

def predictive_model_regression(train_x, train_y, test_x, model_name):
    """
    Predictive model define, train, and test.

    Args:
      - train_x: training features
      - train_y: training labels
      - test_x: testing features
      - model_name: predictive model name

    Returns:
      - test_y_hat: prediction on testing set
    """

    assert model_name in ['linear','mlp','tree_r','ridge','lasso','svr']

    if model_name == 'linear':
        model = LinearRegression()
    elif model_name == 'mlp':
        model = MLPRegressor(hidden_layer_sizes=(50, 50),max_iter=1000)
    elif model_name == 'tree_r':
        model = DecisionTreeRegressor()
    elif model_name == 'ridge':
        model = Ridge()
    elif model_name == 'lasso':
        model = Lasso()
    elif model_name == 'svr':
        model = svm.SVR()


    model.fit(train_x,train_y)
    test_y_hat = model.predict(test_x)

    return test_y_hat
