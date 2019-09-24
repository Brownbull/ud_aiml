# -*- coding: utf-8 -*-
"""
@author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  SLR
"""

def props(cls):   
  return [i for i in cls.__dict__.keys() if i[:1] != '_']
  
#misc libraries
import random
import time
from datetime import datetime

timeStart = time.time()
dtStart = datetime.fromtimestamp(timeStart)
print("Script Start: " + str(dtStart) + "\n" + "-"*25 )

dbg = False
stats = False
# Put numbers of records for sample, otherwise define as False
sample = False
# sample = 30

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
if dbg:
  print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
if dbg:
  print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
if dbg:
  print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
if dbg:
  print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
if dbg:
  print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
if dbg:
 print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
if dbg:
  print("scikit-learn version: {}". format(sklearn.__version__))

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
#%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print("Libraries Initialized \n"+'-'*25)

# Get input datasets
if sample:
  aggregation_enrolls_bin = pd.read_csv('data/AGGREGATION_enrolls_bin.csv', nrows = sample)
else:
  aggregation_enrolls_bin = pd.read_csv('data/AGGREGATION_enrolls_bin.csv')

# Sort values
aggregation_enrolls_bin = aggregation_enrolls_bin.sort_values(by=['Rut'])

# to play with our data we'll create a copy 
data1 = aggregation_enrolls_bin.copy(deep = True) 

# Files columns
data1_x = [
  'NEMScr', 'Ranking', 'LangScr', 'MathScr', 'ScienScr', 'SchoolRegion', 
  'EdTypeCode', 'SchoolType', 'MotherEd', 'CampusStgo', 'PostulationRegular', 
  'S1_INS', 'S1_DRP', 'S1_BAD', 'S1_CVL', 'S1_DID', 'S1_APVD', 'S1_RPVD', 
  'S1_GRD_1TO19', 'S1_GRD_2TO29', 'S1_GRD_3TO39', 
  'S1_GRD_4TO49', 'S1_GRD_5TO59', 'S1_GRD_6TO7', 
  'S1_BEST_GRD', 'S1_WORST_GRD', 'S1_AVG_PERF', 
  'S2_INS', 'S2_DRP', 'S2_BAD', 'S2_CVL', 'S2_DID', 'S2_APVD', 'S2_RPVD', 
  'S2_GRD_1TO19', 'S2_GRD_2TO29', 'S2_GRD_3TO39', 
  'S2_GRD_4TO49', 'S2_GRD_5TO59', 'S2_GRD_6TO7', 
  'S2_BEST_GRD', 'S2_WORST_GRD', 'S2_AVG_PERF', 'S2_VS_S1']

quantitative_cols = [ 
  'NEMScr', 'Ranking', 'LangScr', 'MathScr', 'ScienScr', 
  'S1_INS', 'S1_DRP', 'S1_BAD', 'S1_CVL', 'S1_DID', 'S1_APVD', 'S1_RPVD', 
  'S1_GRD_1TO19', 'S1_GRD_2TO29', 'S1_GRD_3TO39', 
  'S1_GRD_4TO49', 'S1_GRD_5TO59', 'S1_GRD_6TO7', 
  'S1_BEST_GRD', 'S1_WORST_GRD', 'S1_AVG_PERF', 
  'S2_INS', 'S2_DRP', 'S2_BAD', 'S2_CVL', 'S2_DID', 'S2_APVD', 'S2_RPVD', 
  'S2_GRD_1TO19', 'S2_GRD_2TO29', 'S2_GRD_3TO39', 
  'S2_GRD_4TO49', 'S2_GRD_5TO59', 'S2_GRD_6TO7', 
  'S2_BEST_GRD', 'S2_WORST_GRD', 'S2_AVG_PERF', 'S2_VS_S1'
]

quantitative_cols_PSU = [ 
  'NEMScr', 'Ranking', 'LangScr', 'MathScr', 'ScienScr'
]

quantitative_cols_S1 = [ 
  'S1_INS', 'S1_DRP', 'S1_BAD', 'S1_CVL', 'S1_DID', 'S1_APVD', 'S1_RPVD', 
  'S1_GRD_1TO19', 'S1_GRD_2TO29', 'S1_GRD_3TO39', 
  'S1_GRD_4TO49', 'S1_GRD_5TO59', 'S1_GRD_6TO7', 
  'S1_BEST_GRD', 'S1_WORST_GRD', 'S1_AVG_PERF', 
  'S2_VS_S1'
]

quantitative_cols_S2 = [  
  'S2_INS', 'S2_DRP', 'S2_BAD', 'S2_CVL', 'S2_DID', 'S2_APVD', 'S2_RPVD', 
  'S2_GRD_1TO19', 'S2_GRD_2TO29', 'S2_GRD_3TO39', 
  'S2_GRD_4TO49', 'S2_GRD_5TO59', 'S2_GRD_6TO7', 
  'S2_BEST_GRD', 'S2_WORST_GRD', 'S2_AVG_PERF', 'S2_VS_S1'
]


qualitative_cols = [
  'SchoolRegion', 'EdTypeCode', 'SchoolType', 'MotherEd', 'CampusStgo', 'PostulationRegular'
]

# define y variable aka target/outcome
Target = ['Desertor']

data1_qualitative = data1[qualitative_cols + Target]
data1_quantitative = data1[quantitative_cols + Target]

# ML - Start
dataset = aggregation_enrolls_bin.copy(deep = True) 

dataset_cols = [
  'Rut', 
  'NEMScr', 'Ranking', 'LangScr', 'MathScr', 'ScienScr', 'SchoolRegion', 
  'EdTypeCode', 'SchoolType', 'MotherEd', 'CampusStgo', 'PostulationRegular', 
  'S1_INS', 'S1_DRP', 'S1_BAD', 'S1_CVL', 'S1_DID', 'S1_APVD', 'S1_RPVD', 
  'S1_GRD_1TO19', 'S1_GRD_2TO29', 'S1_GRD_3TO39', 
  'S1_GRD_4TO49', 'S1_GRD_5TO59', 'S1_GRD_6TO7', 
  'S1_BEST_GRD', 'S1_WORST_GRD', 'S1_AVG_PERF', 
  'S2_INS', 'S2_DRP', 'S2_BAD', 'S2_CVL', 'S2_DID', 'S2_APVD', 'S2_RPVD', 
  'S2_GRD_1TO19', 'S2_GRD_2TO29', 'S2_GRD_3TO39', 
  'S2_GRD_4TO49', 'S2_GRD_5TO59', 'S2_GRD_6TO7', 
  'S2_BEST_GRD', 'S2_WORST_GRD', 'S2_AVG_PERF', 'S2_VS_S1', 
  'Desertor']

dataset_cols_independent = [
  'Rut', 
  'NEMScr', 'LangScr', 'MathScr', 'ScienScr', 
  'SchoolRegion', 'EdTypeCode', 'SchoolType', 
  'MotherEd', 'CampusStgo', 'PostulationRegular', 
  'S1_DRP', 'S1_BAD', 'S1_CVL', 'S1_APVD', 'S1_RPVD', 
  'S1_GRD_1TO19', 'S1_GRD_2TO29', 'S1_GRD_3TO39', 
  'S1_GRD_4TO49', 'S1_GRD_5TO59', 'S1_GRD_6TO7', 
  'S1_BEST_GRD', 'S1_WORST_GRD',  
  'S2_DRP', 'S2_BAD', 'S2_CVL', 'S2_APVD', 'S2_RPVD', 
  'S2_GRD_1TO19', 'S2_GRD_2TO29', 'S2_GRD_3TO39', 
  'S2_GRD_4TO49', 'S2_GRD_5TO59', 'S2_GRD_6TO7', 
  'S2_BEST_GRD', 'S2_WORST_GRD', 
  'S2_VS_S1', 
  'Desertor']

# Define
X_Qty = ['NEMScr', 'LangScr', 'MathScr', 'ScienScr']
X_Qly = ['SchoolRegion', 'EdTypeCode', 'SchoolType', ]
X1 =  X_Qty + X_Qly
Y1 = 'Desertor'
# Select Features
X = dataset[X1]
# Select Target 
y = dataset[Y1]
 
# Encode Categorical Data
for col in X_Qly:
  data1_dummy = pd.get_dummies(X[[col]], columns=[col])
  # Avoid Dummy variable trap
  dummyCols = data1_dummy.columns.tolist()[1:]
  X[dummyCols] = data1_dummy[dummyCols]
  X.drop(col, axis=1, inplace=True)

# # fixed split
# from sklearn.model_selection import train_test_split
# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)

# # Fitting SLR to the training set
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(train_X, train_y)

# # Predict
# pred_y = regressor.predict(test_X)

# # Evaluate
# from sklearn.metrics import mean_absolute_error
# mean_absolute_error(test_y, pred_y) 

# # Visualising 
# plt.scatter(train_X, train_y, color='red')
# plt.plot(train_X, regressor.predict(train_X), color='blue')
# plt.title(str("'{0}' vs '{1}' SLR prediction (training set)".format(Y1, X1)))
# plt.xlabel("{0}".format(X1))
# plt.ylabel("{0}".format(Y1))
# plt.show()

# ML - End
timeEnd = time.time()
dtEnd = datetime.fromtimestamp(timeEnd)
print('-'*25 + "\nScript End:" + str(dtEnd) + '\n' + '-'*25 + "\nTotal Time: " + str(timeEnd - timeStart) + "\n")
