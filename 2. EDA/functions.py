#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# In[ ]:

def correlation_plot(df, x_label, y_label, x_unit, y_unit):
    '''
    Plots a scatter plot of two features
    ---Paramters---
    df (Pandas DataFrame)
    x_label (str) X axis column label
    y_label (str) Y axis column label
    x_unit (str) X axis units
    y_unit (str) Y axis units
    ---Returns---
    None
    '''
    plt.scatter(df[x_label], df[y_label])
    plt.xlabel(x_label + ' ' + x_unit)
    plt.ylabel(y_label + ' ' + y_unit)
    plt.title(x_label + ' vs '+ y_label)
    plt.show()
    
def jitter(arr):
    '''
    Adds normally distributed jitter to data
    ---Parameters---
    arr (1d np array or pandas series)
    ---Returns---
    arr (1d np array or pandas series) array with jitter
    '''
    return arr + np.random.normal(0, .15, arr.shape)