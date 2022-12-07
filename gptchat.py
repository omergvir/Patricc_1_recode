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
import os

def list_files():
  # Get the list of files in the 'files' folder
  files = os.listdir('files')

  # Return the list of files
  return files

