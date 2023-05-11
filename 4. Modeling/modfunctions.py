#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.stats import randint, uniform

def binning(data):
    '''
    Bins response variable into low [0,6] quality and high [7,10] quality
    ---Parameters---
    data (pandas DataFrame) data to bin
    ---Returns---
    None
    '''
    ranges=[0,6,10]
    group_names=[0, 1]
    data['quality_bin']=pd.cut(data['quality'], bins=ranges, labels=group_names)
    return None  

def threshold_chart(cv, X_train, y_train, threshold=False):
    '''
    Plots a thresholding chart with precision, recall, and f1 score
    ---Parameters---
    cv (sklearn classifier) fitted classifier
    X_train (pandas DataFrame) training data features
    y_train (pandas Series) training data labels
    threshold (boolean or float) whether to put threshold line
    ---Returns---
    None
    '''
    df=pd.DataFrame()
    df['true class labels']=y_train
    df['positive class prob']=cv.predict_proba(X_train)[:,1]
    df.sort_values(by='positive class prob', ascending=True, inplace=True, ignore_index=True)
    f1=[]
    precision=[]
    recall=[]
    for value in df['positive class prob']:
        df['threshold class']= df['positive class prob'] >= value
        y_pred=df['threshold class']
        y_tests=df['true class labels']
        f1.append(f1_score(y_tests, y_pred, average='binary'))
        precision.append(precision_score(y_tests, y_pred))
        recall.append(recall_score(y_tests,y_pred))
    df['f1']=f1
    df['precision']=precision
    df['recall']=recall
    ax = df.plot(x='positive class prob', y=['f1', 'precision','recall'],figsize=(15,9), xticks=np.arange(0,1,.1))
    ax.set_xlabel('Threshold')
    ax.set_title('Precision-Recall vs. Threshold')
    ax.legend(loc = 'lower left')
    if threshold==False:
        pass
    elif threshold== True:
        ax.axvline(0.50, color='black', alpha=0.3)
        ax.text(0.50,.65,'Default Threshold',rotation=90, alpha=0.8, fontsize=16)
    else:
        ax.axvline(threshold, color='black', alpha=0.3)
        ax.text(threshold,.65,'Selected Threshold',rotation=90, alpha=0.8, fontsize=16)
    return None

def plot_feature_importance(importance, names, model_name):
    '''
    Visualizes a bar plot showing feature importance
    ---Parameters---
    importance (np array, pd series, list) coefficients or values of feature importance
    names (np array, pd series, list of str) corresponding feature names
    ---Returns---
    bp (sns.barplot object)
    '''
    data={'feature_names':names, 'feature_importance':importance}
    df=pd.DataFrame(data, columns=['feature_names', 'feature_importance'])
    df['abs']=df['feature_importance'].apply(lambda x: abs(x))
    df.sort_values(by=['abs'], ascending=False,inplace=True)
    plt.figure(figsize=(10,8))
    bp=sns.barplot(data=df, x='feature_importance', y='feature_names')
    plt.tight_layout(pad=1.6)
    plt.title(model_name + ' Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    return bp

def save_plot(plot, filepath):
    '''
    Saves a sns plot as an image
    ---Parameters---
    plot (sns plot object) plot to save
    filepath (str) location to save image
    ---Returns---
    None
    '''
    plot.figure.savefig(filepath)
    return None

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
    cv=GridSearchCV(pipe,param_grid=parameters,cv=kf,scoring='roc_auc', n_jobs=-1)
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
    cv=RandomizedSearchCV(pipe, param_distributions=parameters, cv=kf, scoring='roc_auc', n_iter=50, random_state=123, n_jobs=-1)
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
    y_prob = cv.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend()
    plt.show()
    return None

def record_results(model_name, cv, X_test, y_test, threshold):
    '''
    Records metrics for given classifier
    ---Parameters---
    model_name (str) name of model
    cv (sklearn classifier) fitted classifier
    X_test (pandas DataFrame) test features
    y_test (pandas Series) test target classes
    threshold (float) probability threshold for positive class
    ---Returns---
    list of metrics for classifier 
    '''
    y_pred=(cv.predict_proba(X_test)[:,1]>threshold).astype(int)
    y_prob=cv.predict_proba(X_test)[:,1]
    f1=f1_score(y_test,y_pred, average='binary')
    test_acc=accuracy_score(y_test,y_pred)
    roc=roc_auc_score(y_test, y_prob)
    precision= precision_score(y_test, y_pred)
    recall=recall_score(y_test, y_pred)
    return [model_name, f1, test_acc, roc, precision, recall]

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