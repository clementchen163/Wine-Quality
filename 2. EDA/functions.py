#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# In[ ]:

def correlation_plot(df, x_label, y_label, x_unit, y_unit, graph='both'):
    '''
    Plots 1-2 scatter plots of two features from red and/or white wine
    ---Paramters---
    df (Pandas DataFrame)
    x_label (str) X axis column label
    y_label (str) Y axis column label
    x_unit (str) X axis units
    y_unit (str) Y axis units
    graph (str) graph red, white, or both
    ---Returns---
    None
    '''
    white=df[df['wine_type']=='white']
    red=df[df['wine_type']=='red']
    if graph=='both':
        fig, ax=plt.subplots(1,2, sharex=True, sharey=True)
        
    
        ax[0].scatter(white[x_label], white[y_label],c='yellow')
        ax[0].set_title('White Wine')
        ax[0].set_xlabel(x_label + ' ' + x_unit)
        ax[0].set_ylabel(y_label + ' ' + y_unit)
    
        ax[1].scatter(red[x_label], red[y_label], c='red')
        ax[1].set_title('Red Wine')
        ax[1].set_xlabel(x_label + ' ' + x_unit)
        plt.show()
    elif graph=='white':
        plt.scatter(white[x_label], white[y_label],c='yellow')
        plt.title('White Wine')
        plt.xlabel(x_label + ' ' + x_unit)
        plt.ylabel(y_label + ' ' + y_unit)
    elif graph=='red':
        plt.scatter(red[x_label], red[y_label],c='red')
        plt.title('Red Wine')
        plt.xlabel(x_label + ' ' + x_unit)
        plt.ylabel(y_label + ' ' + y_unit)
    return None
    
def jitter(arr):
    '''
    Adds normally distributed jitter to data
    ---Parameters---
    arr (1d np array or pandas series)
    ---Returns---
    arr (1d np array or pandas series) array with jitter
    '''
    return arr + np.random.normal(0, .15, arr.shape)