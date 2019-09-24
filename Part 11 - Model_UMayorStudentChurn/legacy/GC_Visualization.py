# -*- coding: utf-8 -*-
"""
@author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  Visualization Script
  This script will take Aggregation information and generate visualizations
  input:
    data/AGGREGATION_enrolls_bin.csv
"""

def props(cls):   
  return [i for i in cls.__dict__.keys() if i[:1] != '_']
  
#misc libraries
import random
import time
from datetime import datetime

timeStart = time.time()
dtStart = datetime.fromtimestamp(timeStart)
print("Visualizations Script Start: " + str(dtStart) + "\n" + "-"*25 )

dbg = False
stats = True
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

# Temp file to write just in case
fTemp = open("TempVSL.txt", 'w+')

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
data1_cols = [
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

# define featueres
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

# Visualizations

#Discrete Variable Correlation by Survival using
#group by aka pivot table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
for x in data1_x:
    if data1[x].dtype != 'float64' :
        print('Desertor Correlation by:', x)
        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')
        
#graph distribution of quantitative data
plt.figure(figsize=[16,12])

plt.subplot(231)
plt.hist(x = [data1[data1['Desertor']==1]['NEMScr'], data1[data1['Desertor']==0]['NEMScr']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('NEMScr Histogram by Desertor')
plt.xlabel('NEMScr')
plt.ylabel('# of Students')
plt.legend()

plt.subplot(232)
plt.hist(x = [data1[data1['Desertor']==1]['Ranking'], data1[data1['Desertor']==0]['Ranking']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('Ranking Histogram by Desertor')
plt.xlabel('Ranking')
plt.ylabel('# of Students')
plt.legend()

plt.subplot(234)
plt.hist(x = [data1[data1['Desertor']==1]['LangScr'], data1[data1['Desertor']==0]['LangScr']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('LangScr Histogram by Desertor')
plt.xlabel('LangScr')
plt.ylabel('# of Students')
plt.legend()

plt.subplot(235)
plt.hist(x = [data1[data1['Desertor']==1]['MathScr'], data1[data1['Desertor']==0]['MathScr']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('MathScr Histogram by Desertor')
plt.xlabel('MathScr')
plt.ylabel('# of Students')
plt.legend()

plt.subplot(236)
plt.hist(x = [data1[data1['Desertor']==1]['ScienScr'], data1[data1['Desertor']==0]['ScienScr']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('ScienScr Histogram by Desertor')
plt.xlabel('ScienScr')
plt.ylabel('# of Students')
plt.legend()

#graph distribution of quantitative data
plt.figure(figsize=[16,12])

plt.subplot(231)
plt.hist(x = [data1[data1['Desertor']==1]['S1_DRP'], data1[data1['Desertor']==0]['S1_DRP']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('S1_DRP Histogram by Desertor')
plt.xlabel('S1_DRP')
plt.ylabel('# of Students')
plt.legend()

plt.subplot(232)
plt.hist(x = [data1[data1['Desertor']==1]['S1_BAD'], data1[data1['Desertor']==0]['S1_BAD']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('S1_BAD Histogram by Desertor')
plt.xlabel('S1_BAD')
plt.ylabel('# of Students')
plt.legend()

plt.subplot(233)
plt.hist(x = [data1[data1['Desertor']==1]['S1_CVL'], data1[data1['Desertor']==0]['S1_CVL']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('S1_CVL Histogram by Desertor')
plt.xlabel('S1_CVL')
plt.ylabel('# of Students')
plt.legend()

plt.subplot(234)
plt.hist(x = [data1[data1['Desertor']==1]['S1_DID'], data1[data1['Desertor']==0]['S1_DID']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('S1_DID Histogram by Desertor')
plt.xlabel('S1_DID')
plt.ylabel('# of Students')
plt.legend()

plt.subplot(235)
plt.hist(x = [data1[data1['Desertor']==1]['S1_APVD'], data1[data1['Desertor']==0]['S1_APVD']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('S1_APVD Histogram by Desertor')
plt.xlabel('S1_APVD')
plt.ylabel('# of Students')
plt.legend()

plt.subplot(236)
plt.hist(x = [data1[data1['Desertor']==1]['S1_RPVD'], data1[data1['Desertor']==0]['S1_RPVD']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('S1_RPVD Histogram by Desertor')
plt.xlabel('S1_RPVD')
plt.ylabel('# of Students')
plt.legend()

#graph distribution of quantitative data
plt.figure(figsize=[16,12])

plt.subplot(231)
plt.hist(x = [data1[data1['Desertor']==1]['S1_GRD_1TO19'], data1[data1['Desertor']==0]['S1_GRD_1TO19']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('S1_GRD_1TO19 Histogram by Desertor')
plt.xlabel('S1_GRD_1TO19')
plt.ylabel('# of Students')
plt.legend()

plt.subplot(232)
plt.hist(x = [data1[data1['Desertor']==1]['S1_GRD_2TO29'], data1[data1['Desertor']==0]['S1_GRD_2TO29']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('S1_GRD_2TO29 Histogram by Desertor')
plt.xlabel('S1_GRD_2TO29')
plt.ylabel('# of Students')
plt.legend()

plt.subplot(233)
plt.hist(x = [data1[data1['Desertor']==1]['S1_GRD_3TO39'], data1[data1['Desertor']==0]['S1_GRD_3TO39']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('S1_GRD_3TO39 Histogram by Desertor')
plt.xlabel('S1_GRD_3TO39')
plt.ylabel('# of Students')
plt.legend()

plt.subplot(234)
plt.hist(x = [data1[data1['Desertor']==1]['S1_GRD_4TO49'], data1[data1['Desertor']==0]['S1_GRD_4TO49']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('S1_GRD_4TO49 Histogram by Desertor')
plt.xlabel('S1_GRD_4TO49')
plt.ylabel('# of Students')
plt.legend()

plt.subplot(235)
plt.hist(x = [data1[data1['Desertor']==1]['S1_GRD_5TO59'], data1[data1['Desertor']==0]['S1_GRD_5TO59']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('S1_GRD_5TO59 Histogram by Desertor')
plt.xlabel('S1_GRD_5TO59')
plt.ylabel('# of Students')
plt.legend()

plt.subplot(236)
plt.hist(x = [data1[data1['Desertor']==1]['S1_GRD_6TO7'], data1[data1['Desertor']==0]['S1_GRD_6TO7']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('S1_GRD_6TO7 Histogram by Desertor')
plt.xlabel('S1_GRD_6TO7')
plt.ylabel('# of Students')
plt.legend()

#graph distribution of quantitative data
plt.figure(figsize=[16,12])

plt.subplot(231)
plt.hist(x = [data1[data1['Desertor']==1]['S1_BEST_GRD'], data1[data1['Desertor']==0]['S1_BEST_GRD']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('S1_BEST_GRD Histogram by Desertor')
plt.xlabel('S1_BEST_GRD')
plt.ylabel('# of Students')
plt.legend()

plt.subplot(232)
plt.hist(x = [data1[data1['Desertor']==1]['S1_WORST_GRD'], data1[data1['Desertor']==0]['S1_WORST_GRD']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('S1_WORST_GRD Histogram by Desertor')
plt.xlabel('S1_WORST_GRD')
plt.ylabel('# of Students')
plt.legend()

plt.subplot(233)
plt.hist(x = [data1[data1['Desertor']==1]['S1_AVG_PERF'], data1[data1['Desertor']==0]['S1_AVG_PERF']], 
         stacked=False, color = ['r','g'],label = ['Desertor','Continue'])
plt.title('S1_AVG_PERF Histogram by Desertor')
plt.xlabel('S1_AVG_PERF')
plt.ylabel('# of Students')
plt.legend()

#graph individual features by survival
fig, saxis = plt.subplots(3, 3,figsize=(16,12))

sns.pointplot(x = 'EdTypeCode', y = 'Desertor', data=data1, ax = saxis[0,0])

sns.pointplot(x = 'SchoolType', y = 'Desertor',  data=data1, ax = saxis[1,0])
sns.pointplot(x = 'MotherEd', y = 'Desertor',  data=data1, ax = saxis[1,1])
sns.barplot(x = 'EdTypeCode', y = 'Desertor', order=[0,1,4], data=data1, ax = saxis[1,2])

sns.barplot(x = 'PostulationRegular', y = 'Desertor', order=[1,0], data=data1, ax = saxis[2,0])
sns.pointplot(x = 'SchoolRegion', y = 'Desertor', data=data1, ax = saxis[2,1])
sns.barplot(x = 'SchoolRegion', y = 'Desertor', order=[0,6,9,13], data=data1, ax = saxis[2,2])


#graph distribution of qualitative data: Pclass
#we know class mattered in survival, now let's compare class and a 2nd feature
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))

sns.boxplot(x = 'SchoolRegion', y = 'SchoolType', hue = 'Desertor', data = data1, ax = axis1)
axis1.set_title('SchoolRegion vs SchoolType Desertor Comparison')

sns.violinplot(x = 'SchoolRegion', y = 'PostulationRegular', hue = 'Desertor', data = data1, split = True, ax = axis2)
axis2.set_title('SchoolRegion vs PostulationRegular Desertor Comparison')

sns.boxplot(x = 'SchoolRegion', y ='EdTypeCode', hue = 'Desertor', data = data1, ax = axis3)
axis3.set_title('SchoolRegion vs EdTypeCode Desertor Comparison')


#graph distribution of qualitative data: PostulationRegular
#we know PostulationRegular mattered in survival, now let's compare PostulationRegular and a 2nd feature
fig, qaxis = plt.subplots(1,3,figsize=(14,12))

sns.barplot(x = 'SchoolType', y = 'Desertor', hue = 'PostulationRegular', data=data1, ax = qaxis[0])
axis1.set_title('SchoolType vs PostulationRegular Desertor Comparison')

#correlation heatmap of dataset
def correlation_heatmapBig(df):
    _ , ax = plt.subplots(figsize =(20, 16))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

# correlation_heatmapBig(data1)
correlation_heatmap(data1[quantitative_cols_PSU + Target])
correlation_heatmapBig(data1[quantitative_cols_S1 + Target])
correlation_heatmapBig(data1[quantitative_cols_S2 + Target])
correlation_heatmap(data1_qualitative)

timeEnd = time.time()
dtEnd = datetime.fromtimestamp(timeEnd)
print('-'*25 + "\nVisualizations Script End:" + str(dtEnd) + '\n' + '-'*25 + "\nTotal Time: " + str(timeEnd - timeStart) + "\n")
fTemp.close()