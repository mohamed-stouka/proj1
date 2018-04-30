# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import ensemble
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

from sklearn.metrics import roc_curve,auc
from statsmodels.tools import categorical
import matplotlib.pylab as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing

from urllib.request import urlopen
from urllib.parse import urlencode
from bs4 import BeautifulSoup

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from collections import OrderedDict
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap
import mpl_toolkits

rcParams['figure.figsize'] = 12, 4
