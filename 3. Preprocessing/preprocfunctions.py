#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
def scale_data(df, y):
    '''
    Does train test split and scales data
    ---Parameters---
    df (pandas DataFrame) data with response variable
    y (str) name of column with response variable
    ---Returns---
    train, test (pandas DataFrames) reconstructed train and test dataframes
    '''
    #X and y
    response=df[y]
    X=df.drop(y, axis=1)
    #Train test split
    X_train, X_test, y_train, y_test=train_test_split(X, response, test_size=.2,random_state=123)
    #Record column names and indexes
    X_columns=X.columns
    X_train_index=X_train.index
    X_test_index=X_test.index
    #RobustScaler
    robust_transformer = RobustScaler()
    X_train=robust_transformer.fit_transform(X_train)
    X_test=robust_transformer.transform(X_test)
    #Reconstruct DataFrame
    train=pd.DataFrame(X_train, columns=X_columns, index=X_train_index)
    train[y]=y_train
    
    test=pd.DataFrame(X_test, columns=X_columns, index=X_test_index)
    test[y]=y_test
    
    return train, test