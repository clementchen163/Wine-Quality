#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.stats import randint, uniform


def train_model_GridSearch(X_train, y_train, steps, parameters):
    '''
    Trains a model using GridSearchCV
    ---Parameters---
    X_train (pandas DataFrame) training features
    y_train (pandas Series) training target class
    steps (list) steps used in pipeline
    parameters (dictionary) dictionary of hyperparameters to tune using GridSearchCV
    ---Returns---
    cv (sklearn classifier) fitted model
    '''
    pipe=Pipeline(steps)
    kf=KFold(n_splits=5, shuffle=True, random_state=123)
    cv=GridSearchCV(pipe,param_grid=parameters,cv=kf,scoring='roc_auc')
    cv.fit(X_train,y_train)
    return cv

def train_model_RandomizedSearch(X_train, y_train, steps, parameters):
    '''
    Trains a model using RandomizedSearchCV
    ---Parameters---
    X_train (pandas DataFrame) training features
    y_train (pandas Series) training target classes
    steps (list) steps used in pipeline
    parameters (dictionary) dictionary of hyperparameters to tune using GridSearchCV
    ---Returns---
    cv (sklearn classifier) fitted model
    '''
    pipe=Pipeline(steps)
    kf=KFold(n_splits=5, shuffle=True, random_state=123)
    cv=RandomizedSearchCV(pipe, param_distributions=parameters, cv=kf, scoring='roc_auc', n_iter=50, random_state=123)
    cv.fit(X_train,y_train)
    return cv

def plot_roc(cv, X_test, y_test):
    '''
    Plots roc curve 
    ---Parameters---
    cv (sklearn classifier) fitted classifier
    X_test (pandas DataFrame) test features
    y_test (pandas Series) test target classes
    ---Returns---
    None
    '''
    y_prob = cv.predict_proba(X_test)[:, 0]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label='high')
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend()
    plt.show()
    return None

def record_results(model_name, cv, X_test, y_test):
    '''
    Records metrics for given classifier
    ---Parameters---
    model_name (str) name of model
    cv (sklearn classifier) fitted classifier
    X_test (pandas DataFrame) test features
    y_test (pandas Series) test target classes
    ---Returns---
    list of metrics for classifier 
    '''
    y_pred=cv.predict(X_test)
    y_prob=cv.predict_proba(X_test)[:,1]
    f1=f1_score(y_test,y_pred, average='binary', pos_label='high')
    test_acc=accuracy_score(y_test,y_pred)
    best_score=cv.best_score_
    roc=roc_auc_score(y_test, y_prob)
    return [model_name, f1, test_acc, best_score, roc]

def export_file(export_path, file_name, data):
    '''
    Exports file for later use
    ---Parameters---
    export_path (str) location to save model to
    file_name (str) filename
    data
    ---Returns---
    None
    '''
    filename = export_path + file_name
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    return None