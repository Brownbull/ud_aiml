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
# Temp file to write just in case
fTemp = open("TempMLR.txt", 'w+')

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
X_Qty = ['NEMScr', 'LangScr', 'MathScr', 'ScienScr', 'S1_DRP', 'S1_BAD', 'S1_CVL', 'S1_APVD', 'S1_RPVD', 
  'S1_GRD_1TO19', 'S1_GRD_2TO29', 'S1_GRD_3TO39', 
  'S1_GRD_4TO49', 'S1_GRD_5TO59', 'S1_GRD_6TO7', 
  'S1_BEST_GRD', 'S1_WORST_GRD',  
  'S2_DRP', 'S2_BAD', 'S2_CVL', 'S2_APVD', 'S2_RPVD', 
  'S2_GRD_1TO19', 'S2_GRD_2TO29', 'S2_GRD_3TO39', 
  'S2_GRD_4TO49', 'S2_GRD_5TO59', 'S2_GRD_6TO7', 
  'S2_BEST_GRD', 'S2_WORST_GRD', 
  'S2_VS_S1' ]
X_Qly = ['SchoolRegion', 'EdTypeCode', 'SchoolType', 'MotherEd', 'CampusStgo', 'PostulationRegular']
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

# fixed split
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting MLR
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_X, train_y)

# Predict
pred_y = model.predict(test_X)

# Transform results to true or false
pred_y = (pred_y > 0.5) 
# Transform to numeric
#y_pred = [1 if i > 0.5 else 0 for i in y_pred]
#'Yes' if fruit == 'Apple' else 'No'


#output.to_csv('Titanic_pred.csv', sep=',')

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y, pred_y)

# Accuracy Values
# Defs
TP = cm[0][0]
TN = cm[1][1]
FP = cm[0][1]
FN = cm[1][0]
# Formulas
Accuracy = (TP + TN) / (TP + TN + FP + FN) # 70 80 90 Good
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2 * Precision * Recall / (Precision + Recall) 
print('Accuracy: ', Accuracy)
print('Precision: ', Precision)
print('Recall: ', Recall)
print('F1_Score: ', F1_Score)

# Refine with Backward Elimination
import statsmodels.api as sm
# Add column of 1s to X as b0 constant
X_bin = np.append(arr = np.ones(shape = ( X.shape[0],1)), values = X.astype(int), axis = 1)

# # Evaluate
# from sklearn.metrics import mean_absolute_error
# mean_absolute_error(test_y, pred_y) 
cols2Drop = []

# Features to have in the model

#                 #  coef    std err          t      P>|t|      [0.025      0.975]
opt_X = X_bin[:,[
   0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
  10,11,12,13,14,15,16,17,18,19,
  20,21,22,23,24,25,26,27,28,29,
  30,31,32,33,34,35,36,37,38,39,
  40,41,42,43,44,45,46,47,48]]
# x17            0.0004      0.027      0.014      0.988      -0.053       0.054
# to drop column:17 S2_DRP
cols2Drop.append(16)
opt_X = X_bin[:,[
   0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
  10,11,12,13,14,15,16,18,19,20,
  21,22,23,24,25,26,27,28,29,30,
  31,32,33,34,35,36,37,38,39,40,
  41,42,43,44,45,46,47,48]]
# x11           -0.0014      0.020     -0.067      0.947      -0.041       0.039
# to drop column: 11 
cols2Drop.append(10)
opt_X = X_bin[:,[
   0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
  10,12,13,14,15,16,18,19,20,21,
  22,23,24,25,26,27,28,29,30,31,
  32,33,34,35,36,37,38,39,40,41,
  42,43,44,45,46,47,48]]
# x37            0.0191      0.213      0.090      0.929      -0.400       0.438
# to drop column: 39 
cols2Drop.append(38)
opt_X = X_bin[:,[
   0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
  10,12,13,14,15,16,18,19,20,21,
  22,23,24,25,26,27,28,29,30,31,
  32,33,34,35,36,37,38,40,41,42,
  43,44,45,46,47,48]]
# x34            0.0222      0.115      0.192      0.847      -0.204       0.249
# to drop column: 36 
cols2Drop.append(35)
opt_X = X_bin[:,[
   0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
  10,12,13,14,15,16,18,19,20,21,
  22,23,24,25,26,27,28,29,30,31,
  32,33,34,35,37,38,40,41,42,
  43,44,45,46,47,48]]
# x4          2.497e-05      0.000      0.228      0.820      -0.000       0.000
# to drop column: 4 
cols2Drop.append(3)
opt_X = X_bin[:,[
   0, 1, 2, 3, 5, 6, 7, 8, 9,10,
  12,13,14,15,16,18,19,20,21,22,
  23,24,25,26,27,28,29,30,31,32,
  33,34,35,37,38,40,41,42,43,44,
  45,46,47,48]]
# x3         -8.681e-05      0.000     -0.233      0.816      -0.001       0.001
# # to drop column: 3 
cols2Drop.append(2)
opt_X = X_bin[:,[
   0, 1, 2, 5, 6, 7, 8, 9,10,12,
  13,14,15,16,18,19,20,21,22,23,
  24,25,26,27,28,29,30,31,32,33,
  34,35,37,38,40,41,42,43,44,45,
  46,47,48]]
# x4             0.0068      0.029      0.232      0.817      -0.051       0.064
# to drop column: 6 
cols2Drop.append(5)
opt_X = X_bin[:,[
   0, 1, 2, 5, 7, 8, 9,10,12,13,
  14,15,16,18,19,20,21,22,23,24,
  25,26,27,28,29,30,31,32,33,34,
  35,37,38,40,41,42,43,44,45,46,
  47,48]]
# x4            -0.0013      0.006     -0.227      0.821      -0.013       0.010
# to drop column: 7 
cols2Drop.append(6)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 1, 2, 5, 8, 9,10,12,13,14,
  15,16,18,19,20,21,22,23,24,25,
  26,27,28,29,30,31,32,33,34,35,
  37,38,40,41,42,43,44,45,46,47,
  48]]
# x11            0.0079      0.026      0.310      0.756      -0.042       0.058
# to drop column: 16 
cols2Drop.append(15)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 1, 2, 5, 8, 9,10,12,13,14,
  15,18,19,20,21,22,23,24,25,26,
  27,28,29,30,31,32,33,34,35,37,
  38,40,41,42,43,44,45,46,47,48]]
# x7            -0.0111      0.025     -0.444      0.658      -0.060       0.038
# to drop column: 12 
cols2Drop.append(11)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 1, 2, 5, 8, 9,10,13,14,15,
  18,19,20,21,22,23,24,25,26,27,
  28,29,30,31,32,33,34,35,37,38,
  40,41,42,43,44,45,46,47,48]]
# x5             0.0079      0.018      0.441      0.659      -0.027       0.043
# to drop column: 9 
cols2Drop.append(8)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 1, 2, 5, 8,10,13,14,15,18,
  19,20,21,22,23,24,25,26,27,28,
  29,30,31,32,33,34,35,37,38,40,
  41,42,43,44,45,46,47,48]]
# x9            -0.0046      0.011     -0.432      0.666      -0.025       0.016
# to drop column: 18 
cols2Drop.append(17)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 1, 2, 5, 8,10,13,14,15,19,
  20,21,22,23,24,25,26,27,28,29,
  30,31,32,33,34,35,37,38,40,41,
  42,43,44,45,46,47,48]]
# x20           -0.0116      0.025     -0.469      0.639      -0.060       0.037
# to drop column: 30 
cols2Drop.append(29)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 1, 2, 5, 8,10,13,14,15,19,
  20,21,22,23,24,25,26,27,28,29,
  31,32,33,34,35,37,38,40,41,42,
  43,44,45,46,47,48]]
# x18           -0.0115      0.019     -0.613      0.540      -0.048       0.025
# to drop column: 28 
cols2Drop.append(27)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 1, 2, 5, 8,10,13,14,15,19,
  20,21,22,23,24,25,26,27,29,31,
  32,33,34,35,37,38,40,41,42,43,
  44,45,46,47,48]]
# x17            0.0011      0.026      0.041      0.967      -0.050       0.052
# to drop column: 27 
cols2Drop.append(26)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 1, 2, 5, 8,10,13,14,15,19,
  20,21,22,23,24,25,26,29,31,32,
  33,34,35,37,38,40,41,42,43,44,
  45,46,47,48]]
# x1             0.0001      0.000      0.651      0.515      -0.000       0.000
# to drop column: 1 
cols2Drop.append(0)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 2, 5, 8,10,13,14,15,19,20,
  21,22,23,24,25,26,29,31,32,33,
  34,35,37,38,40,41,42,43,44,45,
  46,47,48]]
# x6            -0.0080      0.010     -0.786      0.432      -0.028       0.012
# to drop column: 14 
cols2Drop.append(13)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 2, 5, 8,10,13,15,19,20,21,
  22,23,24,25,26,29,31,32,33,34,
  35,37,38,40,41,42,43,44,45,46,
  47,48]]
# # x14           -0.0139      0.016     -0.868      0.386      -0.045       0.017
# # to drop column: 26 
cols2Drop.append(25)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 2, 5, 8,10,13,15,19,20,21,
  22,23,24,25,29,31,32,33,34,35,
  37,38,40,41,42,43,44,45,46,47,
  48]]
# x16           -0.0799      0.087     -0.921      0.357      -0.250       0.091
# to drop column: 32 
cols2Drop.append(31)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 2, 5, 8,10,13,15,19,20,21,
  22,23,24,25,29,31,33,34,35,37,
  38,40,41,42,43,44,45,46,47,48]]
# x17           -0.0410      0.046     -0.897      0.370      -0.131       0.049
# to drop column: 34 
cols2Drop.append(33)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 2, 5, 8,10,13,15,19,20,21,
  22,23,24,25,29,31,33,35,37,38,
  40,41,42,43,44,45,46,47,48]]
# x21           -0.0876      0.093     -0.945      0.345      -0.270       0.095
# to drop column: 41 MotherEd_5.
cols2Drop.append(40)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 2, 5, 8,10,13,15,19,20,21,
  22,23,24,25,29,31,33,35,37,38,
  40,42,43,44,45,46,47,48]]
# x24           -0.0452      0.060     -0.750      0.453      -0.163       0.073
# to drop column: 45 MotherEd_11.
cols2Drop.append(44)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 2, 5, 8,10,13,15,19,20,21,
  22,23,24,25,29,31,33,35,37,38,
  40,42,43,44,46,47,48]]
# x21           -0.0340      0.037     -0.929      0.353      -0.106       0.038
# to drop column: 42 MotherEd_7.
cols2Drop.append(41)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 2, 5, 8,10,13,15,19,20,21,
  22,23,24,25,29,31,33,35,37,38,
  40,43,44,46,47,48]]
# x21           -0.0484      0.063     -0.769      0.442      -0.172       0.075
# to drop column: 43 MotherEd_8.
cols2Drop.append(42)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 2, 5, 8,10,13,15,19,20,21,
  22,23,24,25,29,31,33,35,37,38,
  40,44,46,47,48]]
# x22           -0.0392      0.052     -0.760      0.448      -0.140       0.062
# to drop column: 46 CampusStgo_1.
cols2Drop.append(45)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 2, 5, 8,10,13,15,19,20,21,
  22,23,24,25,29,31,33,35,37,38,
  40,44,47,48]]
# x14           -0.0198      0.019     -1.043      0.297      -0.057       0.018
# to drop column: 29 
cols2Drop.append(28)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 2, 5, 8,10,13,15,19,20,21,
  22,23,24,25,31,33,35,37,38,40,
  44,47,48]]
# x7             0.0123      0.011      1.078      0.281      -0.010       0.035
# to drop column: 19 
cols2Drop.append(18)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 2, 5, 8,10,13,15,20,21,22,
  23,24,25,31,33,35,37,38,40,44,
  47,48]]
# x15           -0.0670      0.056     -1.187      0.236      -0.178       0.044
# ********** here is a tie on P value: Personal option
# to drop column: 35 
cols2Drop.append(34)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 2, 5, 8,10,13,15,20,21,22,
  23,24,25,31,33,37,38,40,44,47,48]]
# x1             0.0003      0.000      1.192      0.234      -0.000       0.001
# to drop column: 2 
cols2Drop.append(1)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 5, 8,10,13,15,20,21,22,23,
  24,25,31,33,37,38,40,44,47,48]]
# x6            -0.0212      0.019     -1.120      0.263      -0.058       0.016
# to drop column: 20 
cols2Drop.append(19)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 5, 8,10,13,15,21,22,23,24,
  25,31,33,37,38,40,44,47,48]]
# x5             0.0257      0.021      1.253      0.211      -0.015       0.066
# to drop column: 15 
cols2Drop.append(14)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 5, 8,10,13,21,22,23,24,25,
  31,33,37,38,40,44,47,48]]
# x15           -0.1117      0.083     -1.343      0.180      -0.275       0.052
# to drop column: 44 MotherEd_9.
cols2Drop.append(43)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 5, 8,10,13,21,22,23,24,25,
  31,33,37,38,40,47,48]]
# x7            -0.0301      0.022     -1.376      0.170      -0.073       0.013
# to drop column: 23 
cols2Drop.append(22)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 5, 8,10,13,21,22,24,25,31,
  33,37,38,40,47,48]]
# x8            -0.0035      0.030     -0.116      0.908      -0.063       0.056
# to drop column: 25 
cols2Drop.append(24)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 5, 8,10,13,21,22,24,31,33,
  37,38,40,47,48]]
# x12            0.1200      0.072      1.673      0.095      -0.021       0.261
# to drop column: 40 MotherEd_4.
cols2Drop.append(39)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 5, 8,10,13,21,22,24,31,33,
  37,38,47,48]]
# x3             0.0420      0.024      1.749      0.081      -0.005       0.089
# to drop column: 10 
cols2Drop.append(9)
opt_X = X_bin[:,[
#  0  1  2  3  4  5  6  7  8  9
   0, 5, 8,13,21,22,24,31,33,37,
  38,47,48]]
  
print("Dropped factor from most influence to less influence")
for i, factor in enumerate(X.columns[cols2Drop[::-1]]):
  print(i, ": ",factor)
# Dropped columns:
# 'S1_WORST_GRD', 'S1_GRD_2TO29', 'SchoolType_3', 'EdTypeCode_4',
# 'ScienScr', 'MathScr', 'S1_BAD', 'S1_CVL', 'S1_BEST_GRD',
# 'S1_GRD_3TO39', 'S1_RPVD', 'S2_DRP', 'S2_WORST_GRD', 'S2_GRD_6TO7',
# 'S2_GRD_5TO59', 'NEMScr', 'S1_GRD_5TO59', 'S2_GRD_4TO49',
# 'SchoolRegion_6', 'SchoolRegion_13', 'MotherEd_4.0', 'MotherEd_9.0',
# 'MotherEd_5.0', 'MotherEd_7.0', 'MotherEd_11.0', 'S2_BEST_GRD',
# 'S2_BAD', 'EdTypeCode_1', 'LangScr', 'S2_CVL', 'S1_GRD_6TO7',
# 'MotherEd_8.0', 'S2_GRD_1TO19', 'S2_GRD_3TO39', 'SchoolType_5',
# 'S1_GRD_1TO19'

# Backward Elimination
# 1 - significance 0.05
# 2 - Fit model with all possible predictors
model_OLS = sm.OLS(endog = y, exog = opt_X).fit()
# 3 - consider Feature with highest p-value
fTemp.write(str(model_OLS.summary()))
Pmax = max(model_OLS.pvalues)
print("\nMAX P val :", str(Pmax),"\n")
if Pmax > 0.05:
  print("Colum Drop:", X.columns[[9]])

modelRefined = True

if modelRefined:
  """
  Will use same as 
  opt_X = X_bin[:,[
   0, 5, 8,13,21,22,24,31,33,37,
  38,47,48]]
  But without first('0') and all indexes decreased by 1
  """
  X = X[X.columns[[4, 7,12,20,21,23,30,32,36,37,46,47]]]

  # fixed split
  from sklearn.model_selection import train_test_split
  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)

  # Fitting MLR
  from sklearn.linear_model import LinearRegression
  model = LinearRegression()
  model.fit(train_X, train_y)

  # Predict
  pred_y = model.predict(test_X)

  # Transform results to true or false
  pred_y = (pred_y > 0.5) 
  # Transform to numeric
  #y_pred = [1 if i > 0.5 else 0 for i in y_pred]
  #'Yes' if fruit == 'Apple' else 'No'


  #output.to_csv('Titanic_pred.csv', sep=',')

  # Making the Confusion Matrix
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(test_y, pred_y)

  # Accuracy Values
  # Defs
  TP = cm[0][0]
  TN = cm[1][1]
  FP = cm[0][1]
  FN = cm[1][0]
  # Formulas
  Accuracy = (TP + TN) / (TP + TN + FP + FN) # 70 80 90 Good
  Precision = TP / (TP + FP)
  Recall = TP / (TP + FN)
  F1_Score = 2 * Precision * Recall / (Precision + Recall) 
  print('Accuracy2: ', Accuracy)
  print('Precision2: ', Precision)
  print('Recall2: ', Recall)
  print('F1_Score2: ', F1_Score)

# ML - End
timeEnd = time.time()
dtEnd = datetime.fromtimestamp(timeEnd)
print('-'*25 + "\nScript End:" + str(dtEnd) + '\n' + '-'*25 + "\nTotal Time: " + str(timeEnd - timeStart) + "\n")
fTemp.close()