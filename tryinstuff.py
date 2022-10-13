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

a = [1,2,3,4,5]
b = a[4:]
print(a)
print(b)


