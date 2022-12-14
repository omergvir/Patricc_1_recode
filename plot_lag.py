import os
import pandas as pd
import itertools
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pickle
import math
from scipy import stats
from variables import robot_vs_tablet, granger_condition_list
from matplotlib import pyplot as plt

#df = pd.read_csv('output/all_lag_log_lag30_int03_no_diff.csv')
#df = pd.read_pickle('output/all_chi_log_lag60_int1_w_diff.pkl')
df = pd.read_pickle('output/all_lag_log_lag60_int1_w_diff.pkl')
pick_up = df[df["col2"] == 'robot text:pick up']
#pick_up = df[df["col1"] == 'Parent gaze:props']
pick_up = pick_up.reset_index()
effects = pick_up.col1.unique()

max_lag = len(pick_up['lag'][0])
for effect in effects:
    print(effect)
    mean_list = []
    st_dev_list = []
    pick_up_temp = pick_up[pick_up["col1"] == effect]
    pick_up_temp = pick_up_temp.reset_index()
    for i in range(max_lag):
        temp_lag = []
        for ind in pick_up_temp.index:
            lags_list = pick_up_temp['lag'][ind]
            try:
                a = lags_list[i]
                temp_lag.append(a)
            except:
                pass
        st_dev = np.std(temp_lag)
        mean = np.mean(temp_lag)
        mean_list.append(mean)
        st_dev_list.append(st_dev)
    print(mean_list)
    print(st_dev_list)
    plt.plot(mean_list)

sns.set_style("darkgrid")
plt.legend(labels=effects)
plt.show()