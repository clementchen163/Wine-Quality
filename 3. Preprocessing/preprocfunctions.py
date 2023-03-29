#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler

def scale_data(df, y, dropped=[]):
    '''
    Scales data and reconstructs dataframe with response variable and any non scalable columns
    ---Parameters---
    df (pandas DataFrame) data with response variable
    y (str) name of column with response variable
    dropped (list of str) list of the names of any other columns to be dropped before scaling
    ---Returns---
    data (pandas DataFrame) reconstructed dataframe
    '''
    response=df[y]
    X=df.drop([y]+dropped, axis=1)
    X_columns=X.columns
    drop = df[dropped]
    robust_transformer = RobustScaler()
    X=robust_transformer.fit_transform(X ,response)
    minmax_transformer=MinMaxScaler()
    X=minmax_transformer.fit_transform(X,response)
    data=pd.DataFrame(X, columns=X_columns)
    data[y]=response
    data[dropped]=drop
    return data
    




