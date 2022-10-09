import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import time
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.tsa.stattools import grangercausalitytests
from functools import reduce
from shutil import rmtree




def plus(col1, col2):
        col3 = col1 + col2
        return col3
# initialize data of lists.
data = {'Name': ['Tom', 'nick', 'krish', 'jack'],
        'Age': [20, 21, 19, 18]}

# Create DataFrame
df = pd.DataFrame(data)
cols = df.columns
mat = [[plus(col1, col2) for col2 in cols] for col1 in cols]
print(mat)