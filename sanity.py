import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import time
import math
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.tsa.stattools import grangercausalitytests
from functools import reduce
from shutil import rmtree

def granger(df,col1,col2,maxlag=5):
    c = df.columns[df.nunique() <= 1]  # creates a list of cols with constant values
    # print('these are the constant cols = ', c, 'checking cols = ', col1, col2)
    print(["granger coloumns = ", col1, col2])
    try:
        x = grangercausalitytests(df[[col1, col2]].diff().dropna(),maxlag=maxlag,verbose=False)  # null hypoposis col2 does not granger cause col1
        #x = grangercausalitytests(df[[col1, col2]].dropna(),maxlag=maxlag,verbose=False)  # null hypoposis col2 does not granger cause col1

        lags = list(range(1,maxlag+1))
        lag_pv = np.array([x[lag][0]['ssr_chi2test'][1] for lag in lags])
        min_pv = min(lag_pv)
        lag_chi2v = np.array([x[lag][0]['ssr_chi2test'][0] for lag in lags])
        best_chi2v = max(lag_chi2v)
        max_index = lag_chi2v.argmax(axis=0)
        #best_index = lag_pv.argmin(axis=0)
        best_pv = lag_pv[max_index]
        #best_chi2v = lag_chi2v[best_index]
        best_lag = np.array(lags)[lag_pv == best_pv] if len(lag_pv == best_pv) == 1 else np.array(lags)[lag_pv == best_pv][0]
        if min_pv < 0.05:
            min_pv = 1
        else:
            min_pv = 0

    except Exception as e:
        best_lag = 100
        best_pv = math.nan
        best_chi2v = math.nan
        lag_chi2v = math.nan
        lag_pv = math.nan
        min_pv = math.nan
        print(e)
    print(best_chi2v)
    # return([best_lag,best_pv])
    return(best_chi2v, min_pv, lag_pv, lag_chi2v)



# create a list of random numbers with a length 0f 300
random_list = np.random.randint(0, 100, 300)
# create a list of random numbers with a length of 10
random_list2 = np.random.randint(0, 100, 15)
#create a list that is random_list2 and after that random_list
random_list3 = random_list2.tolist() + random_list.tolist()
#create a new list that containts only the first 300 elements of random_list
random_list4 = random_list3[:300]
# convert random_list4 to a numpy array
random_list4 = np.array(random_list4)
# add random noise to random_list4
random_list4 = random_list4 + np.random.normal(0, 1, 300)
# add random noise to random_list
random_list = random_list + np.random.normal(0, 1, 300)

col1 = random_list4
col2 = random_list
coll1 = ''
maxlag = 10
#create a dataframe with the two columns
df = pd.DataFrame({'col1': col1, 'col2': col2})
# plot the two columns of df
df.plot()
# show the plot
plt.show()
a,b,c,d = granger(df,'col1','col2',maxlag=30)
# plot c and d
#create a plot with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1)
# on the top plot c
ax1.plot(c)
#on the bottom plot d
ax2.plot(d)
# show the plot
plt.show()
