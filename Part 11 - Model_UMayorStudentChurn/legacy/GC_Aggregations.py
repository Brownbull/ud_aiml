# -*- coding: utf-8 -*-
"""
@author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  AGGREGATIONS Script
  This script will take filters csv information of students and their grades to
  generate new files with processed aggregation information per student
  input:
    data/FILTERS_Enrolls_bin.csv
    data/FILTERS_Enrolls_calc.csv
    data/FILTERS_Grades_bin.csv
    data/FILTERS_Grades_calc.csv
"""

def props(cls):   
  return [i for i in cls.__dict__.keys() if i[:1] != '_']
  
#misc libraries
import random
import time
from datetime import datetime

timeStart = time.time()
dtStart = datetime.fromtimestamp(timeStart)
print("AGGREGATIONS Script Start: " + str(dtStart) + "\n" + "-"*25 )

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
fTemp = open("TempAGG.txt", 'w+')

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
  enrolls_bin = pd.read_csv('data/FILTERS_Enrolls_bin.csv', nrows = sample)
  enrolls_calc = pd.read_csv('data/FILTERS_Enrolls_calc.csv', nrows = sample)
  grades_bin = pd.read_csv('data/FILTERS_Grades_bin.csv', nrows = sample)
  grades_calc = pd.read_csv('data/FILTERS_Grades_calc.csv', nrows = sample)
else:
  enrolls_bin = pd.read_csv('data/FILTERS_Enrolls_bin.csv')
  enrolls_calc = pd.read_csv('data/FILTERS_Enrolls_calc.csv')
  grades_bin = pd.read_csv('data/FILTERS_Grades_bin.csv')
  grades_calc = pd.read_csv('data/FILTERS_Grades_calc.csv')

# Rut id format
enrolls_bin['Rut'] = pd.to_numeric(enrolls_bin['Rut'])
grades_bin['Rut'] = pd.to_numeric(grades_bin['Rut'])

# Sort values
enrolls_bin = enrolls_bin.sort_values(by=['Rut', 'EntryYear'])
grades_bin = grades_bin.sort_values(by=['Rut', 'Year', 'Period'])

# to play with our data we'll create a copy 
enrolls_data_bin = enrolls_bin.copy(deep = True) 
enrolls_data_calc = enrolls_calc.copy(deep = True) 
grades_data_bin = grades_bin.copy(deep = True) 
grades_data_calc = grades_calc.copy(deep = True) 

# Files columns
enrolls_data_bin_cols = [
  'Rut', 'EntryYear', 'NEMScr', 'Ranking', 'LangScr', 'MathScr', 'ScienScr', 
  'SchoolRegion', 'EdTypeCode', 'SchoolType', 'MotherEd', 'CampusStgo', 
  'PostulationRegular', 'Desertor']
grades_data_bin_cols = [
  'Rut', 'Year', 'Period', 'SubjStatus', 'Performance']

#cleanup rare cases
stat_min = 15 # while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
varCounts = (enrolls_data_bin['SchoolRegion'].value_counts() < stat_min)
enrolls_data_bin['SchoolRegion'] = enrolls_data_bin['SchoolRegion'].apply(lambda x: '0' if varCounts.loc[x] == True else x)
varCounts = (enrolls_data_bin['EdTypeCode'].value_counts() < stat_min)
enrolls_data_bin['EdTypeCode'] = enrolls_data_bin['EdTypeCode'].apply(lambda x: '0' if varCounts.loc[x] == True else x)
varCounts = (enrolls_data_bin['MotherEd'].value_counts() < stat_min)
enrolls_data_bin['MotherEd'] = enrolls_data_bin['MotherEd'].apply(lambda x: '0' if varCounts.loc[x] == True else x)
varCounts = (enrolls_data_bin['SchoolType'].value_counts() < stat_min)
enrolls_data_bin['SchoolType'] = enrolls_data_bin['SchoolType'].apply(lambda x: '0' if varCounts.loc[x] == True else x)

# Calculate Aggregations
Population = []
Studnt = {
  "Rut" : 0, "Year": 0,
  "S1_INS": 0, "S1_DRP": 0, "S1_BAD": 0, "S1_CVL": 0, 
  "S1_DID": 0, "S1_APVD": 0, "S1_RPVD": 0, 
  "S1_GRD_1TO19": 0, "S1_GRD_2TO29": 0, "S1_GRD_3TO39": 0, 
  "S1_GRD_4TO49": 0, "S1_GRD_5TO59": 0, "S1_GRD_6TO7": 0, 
  "S1_BEST_GRD": 0, "S1_WORST_GRD": 0, "S1_AVG_PERF": 0, 

  "S2_INS": 0, "S2_DRP": 0, "S2_BAD": 0, "S2_CVL": 0, 
  "S2_DID": 0, "S2_APVD": 0, "S2_RPVD": 0, 
  "S2_GRD_1TO19": 0, "S2_GRD_2TO29": 0, "S2_GRD_3TO39": 0, 
  "S2_GRD_4TO49": 0, "S2_GRD_5TO59": 0, "S2_GRD_6TO7": 0, 
  "S2_BEST_GRD": 0, "S2_WORST_GRD": 0, "S2_AVG_PERF": 0, 

  "S2_VS_S1": 0}

for idx, row in grades_data_bin.iterrows():
  if enrolls_data_bin['Rut'].isin([row['Rut']]).count() > 0:
    if Studnt["Rut"] == 0:
      Studnt["Rut"] = grades_data_bin.loc[idx,'Rut']
      Studnt["Year"] = grades_data_bin.loc[idx,'Year']
    elif Studnt["Rut"] != grades_data_bin.loc[idx,'Rut']:
      # print(Studnt)
      Studnt["S1_DID"] = Studnt["S1_INS"] - Studnt["S1_DRP"] - Studnt["S1_BAD"] - Studnt["S1_CVL"]
      Studnt["S1_AVG_PERF"] = 0 if Studnt["S1_DID"] == 0 else round(Studnt["S1_AVG_PERF"] / Studnt["S1_DID"],2)

      Studnt["S2_DID"] = Studnt["S2_INS"] - Studnt["S2_DRP"] - Studnt["S2_BAD"] - Studnt["S2_CVL"]
      Studnt["S2_AVG_PERF"] = 0 if Studnt["S2_DID"] == 0 else round(Studnt["S2_AVG_PERF"] / Studnt["S2_DID"],2)
      Studnt["S2_VS_S1"] = Studnt["S2_AVG_PERF"] - Studnt["S1_AVG_PERF"]
      Population.append(Studnt)
      Studnt = {
        "Rut" : grades_data_bin.loc[idx,'Rut'], 
        "Year": grades_data_bin.loc[idx,'Year'],
        "S1_INS": 0, "S1_DRP": 0, "S1_BAD": 0, "S1_CVL": 0, 
        "S1_DID": 0, "S1_APVD": 0, "S1_RPVD": 0, 
        "S1_GRD_1TO19": 0, "S1_GRD_2TO29": 0, "S1_GRD_3TO39": 0, 
        "S1_GRD_4TO49": 0, "S1_GRD_5TO59": 0, "S1_GRD_6TO7": 0, 
        "S1_BEST_GRD": 0, "S1_WORST_GRD": 0, "S1_AVG_PERF": 0, 

        "S2_INS": 0, "S2_DRP": 0, "S2_BAD": 0, "S2_CVL": 0, 
        "S2_DID": 0, "S2_APVD": 0, "S2_RPVD": 0, 
        "S2_GRD_1TO19": 0, "S2_GRD_2TO29": 0, "S2_GRD_3TO39": 0, 
        "S2_GRD_4TO49": 0, "S2_GRD_5TO59": 0, "S2_GRD_6TO7": 0, 
        "S2_BEST_GRD": 0, "S2_WORST_GRD": 0, "S2_AVG_PERF": 0, 

        "S2_VS_S1": 0}
    
    if grades_data_bin.loc[idx,'Year'] == Studnt["Year"]:
      # Semester 1
      if grades_data_bin.loc[idx,'Period'] == 1:
        Studnt["S1_INS"] += 1
        if 1.0 <= grades_data_bin.loc[idx,'Performance'] <= 7.0:
          if grades_data_bin.loc[idx,'Performance'] < 4.0:
            Studnt["S1_RPVD"] += 1
            if grades_data_bin.loc[idx,'Performance'] >= 3.0:
              Studnt["S1_GRD_3TO39"] += 1
            elif grades_data_bin.loc[idx,'Performance'] >= 2.0:
              Studnt["S1_GRD_2TO29"] += 1
            elif grades_data_bin.loc[idx,'Performance'] >= 1.0:
              Studnt["S1_GRD_1TO19"] += 1
          else:
            Studnt["S1_APVD"] += 1
            if grades_data_bin.loc[idx,'Performance'] < 5.0:
              Studnt["S1_GRD_4TO49"] += 1
            elif grades_data_bin.loc[idx,'Performance'] < 6.0:
              Studnt["S1_GRD_5TO59"] += 1
            elif grades_data_bin.loc[idx,'Performance'] <= 7.0:
              Studnt["S1_GRD_6TO7"] += 1
          
          if Studnt["S1_BEST_GRD"] == 0:
            Studnt["S1_BEST_GRD"] = grades_data_bin.loc[idx,'Performance']
          elif Studnt["S1_BEST_GRD"] < grades_data_bin.loc[idx,'Performance']:
            Studnt["S1_BEST_GRD"] = grades_data_bin.loc[idx,'Performance']

          if Studnt["S1_WORST_GRD"] == 0:
            Studnt["S1_WORST_GRD"] = grades_data_bin.loc[idx,'Performance']
          elif Studnt["S1_WORST_GRD"] > grades_data_bin.loc[idx,'Performance']:
            Studnt["S1_WORST_GRD"] = grades_data_bin.loc[idx,'Performance']

          Studnt["S1_AVG_PERF"] += grades_data_bin.loc[idx,'Performance']

        elif grades_data_bin.loc[idx,'Performance'] == 91: #'DRP'
          Studnt["S1_DRP"] += 1
        elif grades_data_bin.loc[idx,'Performance'] == 92: #'BAD'
          Studnt["S1_BAD"] += 1
        elif grades_data_bin.loc[idx,'Performance'] == 93: #'CVL'
          Studnt["S1_CVL"] += 1
      # Semester 2   
      elif grades_data_bin.loc[idx,'Period'] == 2:
        Studnt["S2_INS"] += 1
        if 1.0 <= grades_data_bin.loc[idx,'Performance'] <= 7.0:
          if grades_data_bin.loc[idx,'Performance'] < 4.0:
            Studnt["S2_RPVD"] += 1
            if grades_data_bin.loc[idx,'Performance'] >= 3.0:
              Studnt["S2_GRD_3TO39"] += 1
            elif grades_data_bin.loc[idx,'Performance'] >= 2.0:
              Studnt["S2_GRD_2TO29"] += 1
            elif grades_data_bin.loc[idx,'Performance'] >= 1.0:
              Studnt["S2_GRD_1TO19"] += 1
          else:
            Studnt["S2_APVD"] += 1
            if grades_data_bin.loc[idx,'Performance'] < 5.0:
              Studnt["S2_GRD_4TO49"] += 1
            elif grades_data_bin.loc[idx,'Performance'] < 6.0:
              Studnt["S2_GRD_5TO59"] += 1
            elif grades_data_bin.loc[idx,'Performance'] <= 7.0:
              Studnt["S2_GRD_6TO7"] += 1
          
          if Studnt["S2_BEST_GRD"] == 0:
            Studnt["S2_BEST_GRD"] = grades_data_bin.loc[idx,'Performance']
          elif Studnt["S2_BEST_GRD"] < grades_data_bin.loc[idx,'Performance']:
            Studnt["S2_BEST_GRD"] = grades_data_bin.loc[idx,'Performance']

          if Studnt["S2_WORST_GRD"] == 0:
            Studnt["S2_WORST_GRD"] = grades_data_bin.loc[idx,'Performance']
          elif Studnt["S2_WORST_GRD"] > grades_data_bin.loc[idx,'Performance']:
            Studnt["S2_WORST_GRD"] = grades_data_bin.loc[idx,'Performance']

          Studnt["S2_AVG_PERF"] += grades_data_bin.loc[idx,'Performance']

        elif grades_data_bin.loc[idx,'Performance'] == 91: #'DRP'
          Studnt["S2_DRP"] += 1
        elif grades_data_bin.loc[idx,'Performance'] == 92: #'BAD'
          Studnt["S2_BAD"] += 1
        elif grades_data_bin.loc[idx,'Performance'] == 93: #'CVL'
          Studnt["S2_CVL"] += 1
  else:
    print("NOT Found: ", row['Rut'])

# Append Last student
Studnt["S1_DID"] = Studnt["S1_INS"] - Studnt["S1_DRP"] - Studnt["S1_BAD"] - Studnt["S1_CVL"]
Studnt["S1_AVG_PERF"] = 0 if Studnt["S1_DID"] == 0 else round(Studnt["S1_AVG_PERF"] / Studnt["S1_DID"],2)

Studnt["S2_DID"] = Studnt["S2_INS"] - Studnt["S2_DRP"] - Studnt["S2_BAD"] - Studnt["S2_CVL"]
Studnt["S2_AVG_PERF"] = 0 if Studnt["S2_DID"] == 0 else round(Studnt["S2_AVG_PERF"] / Studnt["S2_DID"],2)
Studnt["S2_VS_S1"] = Studnt["S2_AVG_PERF"] - Studnt["S1_AVG_PERF"]
Population.append(Studnt)

# Add Aggregations to enrolls
for idx, row in enrolls_data_bin.iterrows():
  Studnt = list(filter(lambda Studnt: Studnt['Rut'] == row['Rut'], Population))
  if Studnt:
    enrolls_data_bin.loc[idx,'S1_INS'] = Studnt[0]['S1_INS']
    enrolls_data_bin.loc[idx,'S1_DRP'] = Studnt[0]['S1_DRP']
    enrolls_data_bin.loc[idx,'S1_BAD'] = Studnt[0]['S1_BAD']
    enrolls_data_bin.loc[idx,'S1_CVL'] = Studnt[0]['S1_CVL']
    enrolls_data_bin.loc[idx,'S1_DID'] = Studnt[0]['S1_DID']
    enrolls_data_bin.loc[idx,'S1_APVD'] = Studnt[0]['S1_APVD']
    enrolls_data_bin.loc[idx,'S1_RPVD'] = Studnt[0]['S1_RPVD']
    enrolls_data_bin.loc[idx,'S1_GRD_1TO19'] = Studnt[0]['S1_GRD_1TO19']
    enrolls_data_bin.loc[idx,'S1_GRD_2TO29'] = Studnt[0]['S1_GRD_2TO29']
    enrolls_data_bin.loc[idx,'S1_GRD_3TO39'] = Studnt[0]['S1_GRD_3TO39']
    enrolls_data_bin.loc[idx,'S1_GRD_4TO49'] = Studnt[0]['S1_GRD_4TO49']
    enrolls_data_bin.loc[idx,'S1_GRD_5TO59'] = Studnt[0]['S1_GRD_5TO59']
    enrolls_data_bin.loc[idx,'S1_GRD_6TO7'] = Studnt[0]['S1_GRD_6TO7']
    enrolls_data_bin.loc[idx,'S1_BEST_GRD'] = Studnt[0]['S1_BEST_GRD']
    enrolls_data_bin.loc[idx,'S1_WORST_GRD'] = Studnt[0]['S1_WORST_GRD']
    enrolls_data_bin.loc[idx,'S1_AVG_PERF'] = Studnt[0]['S1_AVG_PERF']

    enrolls_data_bin.loc[idx,'S2_INS'] = Studnt[0]['S2_INS']
    enrolls_data_bin.loc[idx,'S2_DRP'] = Studnt[0]['S2_DRP']
    enrolls_data_bin.loc[idx,'S2_BAD'] = Studnt[0]['S2_BAD']
    enrolls_data_bin.loc[idx,'S2_CVL'] = Studnt[0]['S2_CVL']
    enrolls_data_bin.loc[idx,'S2_DID'] = Studnt[0]['S2_DID']
    enrolls_data_bin.loc[idx,'S2_APVD'] = Studnt[0]['S2_APVD']
    enrolls_data_bin.loc[idx,'S2_RPVD'] = Studnt[0]['S2_RPVD']
    enrolls_data_bin.loc[idx,'S2_GRD_1TO19'] = Studnt[0]['S2_GRD_1TO19']
    enrolls_data_bin.loc[idx,'S2_GRD_2TO29'] = Studnt[0]['S2_GRD_2TO29']
    enrolls_data_bin.loc[idx,'S2_GRD_3TO39'] = Studnt[0]['S2_GRD_3TO39']
    enrolls_data_bin.loc[idx,'S2_GRD_4TO49'] = Studnt[0]['S2_GRD_4TO49']
    enrolls_data_bin.loc[idx,'S2_GRD_5TO59'] = Studnt[0]['S2_GRD_5TO59']
    enrolls_data_bin.loc[idx,'S2_GRD_6TO7'] = Studnt[0]['S2_GRD_6TO7']
    enrolls_data_bin.loc[idx,'S2_BEST_GRD'] = Studnt[0]['S2_BEST_GRD']
    enrolls_data_bin.loc[idx,'S2_WORST_GRD'] = Studnt[0]['S2_WORST_GRD']
    enrolls_data_bin.loc[idx,'S2_AVG_PERF'] = Studnt[0]['S2_AVG_PERF']

    enrolls_data_bin.loc[idx,'S2_VS_S1'] = Studnt[0]['S2_VS_S1']
  else:
    print("NOT Found: ", row['Rut'])

# Drop source columns after Aggregations
# enrolls_data_bin_drop_cols = []
# enrolls_data_bin.drop(enrolls_data_bin_drop_cols, axis=1, inplace = True)

print("-"*25 + "\nAggregations Created\n" + "-"*25 )

# After Aggregtions Creation
if stats:
  # info
  enrolls_data_bin.info()
  print("-"*10)
  # describe
  enrolls_data_bin.describe(include = 'all').to_csv("reports/AGGREGATIONS_enrolls_data_bin_desc.csv")
  # nulls
  print('enrolls_data_bin columns with null values:\n', enrolls_data_bin.isnull().sum())
  print("-"*10)

# Create Aggregations bin CSV Files
grades_data_bin_cols = [
  "Rut", "NEMScr", "Ranking", "LangScr", "MathScr", "ScienScr", 
  "SchoolRegion", "EdTypeCode", "SchoolType", "MotherEd", "CampusStgo", 
  "PostulationRegular", 
  "S1_INS", "S1_DRP", "S1_BAD", "S1_CVL", "S1_DID", "S1_APVD", "S1_RPVD", 
  "S1_GRD_1TO19", "S1_GRD_2TO29", "S1_GRD_3TO39", 
  "S1_GRD_4TO49", "S1_GRD_5TO59", "S1_GRD_6TO7", 
  "S1_BEST_GRD", "S1_WORST_GRD", "S1_AVG_PERF", 
  "S2_INS", "S2_DRP", "S2_BAD", "S2_CVL", "S2_DID", "S2_APVD", "S2_RPVD", 
  "S2_GRD_1TO19", "S2_GRD_2TO29", "S2_GRD_3TO39", 
  "S2_GRD_4TO49", "S2_GRD_5TO59", "S2_GRD_6TO7", 
  "S2_BEST_GRD", "S2_WORST_GRD", "S2_AVG_PERF", 
  "S2_VS_S1", "Desertor"]
enrolls_data_bin = enrolls_data_bin[grades_data_bin_cols]
enrolls_data_bin.to_csv("data/AGGREGATION_enrolls_bin.csv", index=False)

timeEnd = time.time()
dtEnd = datetime.fromtimestamp(timeEnd)
print('-'*25 + "\nAGGREGATIONS Script End:" + str(dtEnd) + '\n' + '-'*25 + "\nTotal Time: " + str(timeEnd - timeStart) + "\n")
fTemp.close()