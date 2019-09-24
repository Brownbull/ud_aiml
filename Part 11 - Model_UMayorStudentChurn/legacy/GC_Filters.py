# -*- coding: utf-8 -*-
"""
@author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  FILTERS Script
  This script will take raw csv information of students and their grades to
  generate new files with processed information
  input:
    data/Enrolls.csv
    data/Grades.csv
"""
from lib.base_modules import *

timeStart = time.time()
dtStart = datetime.fromtimestamp(timeStart)
print("FILTERS Script Start: " + str(dtStart) + "\n" + "-"*25 )

dbg = False
stats = True
# Put numbers of records for sample, otherwise define as False
sample = False

# Temp file to write just in case
fTemp = open("TempFIL.txt", 'w+')

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
  enrolls_raw = pd.read_csv("data/Enrolls.csv", nrows = sample)
  grades_raw = pd.read_csv("data/Grades.csv", nrows = sample)
else:
  enrolls_raw = pd.read_csv("data/Enrolls.csv")
  grades_raw = pd.read_csv("data/Grades.csv")

# Rename columns
enrolls_raw.columns = ['EntryYear', 'TypeId', 'Rut', 'PlanId', 'DemreCode', 'Career', 'Campus', 'PostulationType', 'EntryType', 'NEM', 'NEMScr', 'Ranking', 'LangScr', 'MathScr', 'ScienScr', 'HistScr', 'PrefUM', 'PrefDemre', 'SchoolCity', 'SchoolRegion', 'EdTypeCode', 'EdType', 'SchoolType', 'MotherEd']
grades_raw.columns = ['Rut', 'StudyPlan', 'PlanName', 'Subject', 'SubjName', 'EnrollReason', 'ReasonDesc', 'SubjStatus', 'Year', 'Period', 'SubjValidate', 'Section', 'Grade', 'SubjHomologate' ]

# Remove last character in Rut
enrolls_raw['Rut'] = enrolls_raw['Rut'].astype(str).str[:-1]
grades_raw['Rut'] = grades_raw['Rut'].astype(str).str[:-1]

# Sort values
enrolls_raw = enrolls_raw.sort_values(by=['Rut', 'EntryYear'])
grades_raw = grades_raw.sort_values(by=['Rut', 'Year', 'Period', 'Grade', 'SubjStatus'])

# Drop duplicates
enrolls_raw = enrolls_raw.drop_duplicates(subset=['EntryYear', 'TypeId', 'Rut', 'DemreCode', 'Career', 'Campus', 'PostulationType', 'NEM', 'NEMScr', 'Ranking', 'LangScr', 'MathScr', 'ScienScr', 'HistScr', 'PrefUM', 'PrefDemre', 'SchoolRegion', 'EdTypeCode', 'EdType', 'SchoolType', 'MotherEd'], keep='last')
grades_raw = grades_raw.drop_duplicates(subset=['Rut', 'Subject', 'SubjName', 'EnrollReason', 'ReasonDesc', 'SubjStatus', 'Year', 'Period', 'SubjValidate', 'Section', 'Grade', 'SubjHomologate' ], keep='last')

# to play with our data we'll create a copy 
enrolls_data = enrolls_raw.copy(deep = True) 
grades_data = grades_raw.copy(deep = True) 

if stats:
  # info
  enrolls_data.info()
  print("-"*10) 
  grades_data.info()
  print("-"*10)
  # describe
  enrolls_data.describe(include = 'all').to_csv("reports/FILTERS_enrolls_desc.csv")
  grades_data.describe(include = 'all').to_csv("reports/FILTERS_grades_desc.csv")
  # nulls
  print('enrolls_data columns with null values:\n', enrolls_data.isnull().sum())
  print("-"*10)
  print('grades_data columns with null values:\n', grades_data.isnull().sum())
  print("-"*10)

# Drop non significant columns
enrolls_drop_cols = ['TypeId','PlanId','DemreCode','Career','EntryType','NEM','HistScr','PrefUM','PrefDemre','SchoolCity','EdType']
enrolls_data.drop(enrolls_drop_cols, axis=1, inplace = True)
grades_drop_cols = ['StudyPlan','PlanName','Subject','SubjName','EnrollReason','ReasonDesc','Section']
grades_data.drop(grades_drop_cols, axis=1, inplace = True)

# Drop non deducible rows
grades_data = grades_data.dropna(subset=['Period'])

print("-"*25 + "\nColumns and Rows Drop\n" + "-"*25 )

# After Delete Cols
if stats:
  # info
  enrolls_data.info()
  print("-"*10)
  grades_data.info()
  print("-"*10)
  # describe
  enrolls_data.describe(include = 'all').to_csv("reports/FILTERS_enrolls_desc_afterDelete.csv")
  grades_data.describe(include = 'all').to_csv("reports/FILTERS_grades_desc_afterDelete.csv")
  # nulls
  print('enrolls_data columns with null values:\n', enrolls_data.isnull().sum())
  print("-"*10)
  print('grades_data columns with null values:\n', grades_data.isnull().sum())
  print("-"*10)

# Filling Data
# Enrolls
"""
Enrolls Fills
Ranking : median
NEMScr : median
LangScr: median
MathScr: median
ScienScr: median
SchoolRegion: mode
EdTypeCode: mode
SchoolType: mode
MotherEd: mode
"""
enrolls_data['Ranking'].fillna(enrolls_data['Ranking'].median(), inplace = True)
enrolls_data['NEMScr'].fillna(enrolls_data['NEMScr'].median(), inplace = True)
enrolls_data['LangScr'].fillna(enrolls_data['LangScr'].median(), inplace = True)
enrolls_data['MathScr'].fillna(enrolls_data['MathScr'].median(), inplace = True)
enrolls_data['ScienScr'].fillna(enrolls_data['ScienScr'].median(), inplace = True)

enrolls_data['SchoolRegion'].fillna(enrolls_data['SchoolRegion'].mode()[0], inplace = True)
enrolls_data['EdTypeCode'].fillna(enrolls_data['EdTypeCode'].mode()[0], inplace = True)
enrolls_data['SchoolType'].fillna(enrolls_data['SchoolType'].mode()[0], inplace = True)
enrolls_data['MotherEd'].fillna(enrolls_data['MotherEd'].mode()[0], inplace = True)

print("-"*25 + "\nColumns Filled\n" + "-"*25 )

# Feature Engineering

# Target : Desertor
# 'y' variable aka target/outcome
# Student which does not have subject after 1 year are considered deserter
grades_maxYear = grades_data.groupby('Rut').max()['Year']
grades_minYear = grades_data.groupby('Rut').min()['Year']

for idx, row in enrolls_data.iterrows():
  if row['Rut'] in grades_maxYear.keys():
    if sample:
      print("Rut: " + str(row['Rut']) + " Max: " + str(grades_maxYear[row['Rut']]) + " Min: " + str(grades_minYear[row['Rut']]) )
    enrolls_data.loc[idx,'Desertor'] = 1 if grades_maxYear[row['Rut']] - grades_minYear[row['Rut']] == 0 else 0
  else:
    print("Rut NOT found: " + str(row['Rut']))
    enrolls_data.loc[idx,'Desertor'] = -1

# Create Filtered calc CSV Files -  to use on graphs
grades_data_calc_cols = [
  'Rut','Year','Period','SubjStatus','SubjHomologate','SubjValidate','Grade']
grades_data = grades_data[grades_data_calc_cols]
grades_data.to_csv("data/FILTERS_Grades_calc.csv", index=False)

enrolls_data_calc_cols = [ 
  'Rut', 'EntryYear', 'NEMScr', 'Ranking', 'LangScr', 'MathScr', 'ScienScr', 
  'SchoolRegion', 'EdTypeCode', 'SchoolType', 'MotherEd', 'Campus', 
  'PostulationType', 'Desertor']
enrolls_data = enrolls_data[enrolls_data_calc_cols]
enrolls_data.to_csv("data/FILTERS_Enrolls_calc.csv", index=False)

# Features 
# Enrolls: CampusStgo, PostulationRegular, SchoolRegion, EdTypeCode, SchoolType
"""
CampusStgo: from Campus
PostulationRegular : from PostulationType
PostulationType: Replace roman per numbers
EdTypeCode: Replace codes per numbers
SchoolType: Replace codes per numbers
"""
for idx, row in enrolls_data.iterrows():
  # CampusStgo
  if enrolls_data.loc[idx,'Campus'][0] in ['S', 's']:
    enrolls_data.loc[idx,'CampusStgo'] = 1
  else:
    enrolls_data.loc[idx,'CampusStgo'] = 0
  # PostulationRegular
  if enrolls_data.loc[idx,'PostulationType'][0] in ['R', 'r']:
    enrolls_data.loc[idx,'PostulationRegular'] = 1
  else:
    enrolls_data.loc[idx,'PostulationRegular'] = 0
  # SchoolRegion
  if enrolls_data['SchoolRegion'].dtype != np.float64: 
  # added for sample, 10 rows will case np.float64 tpe which does not have isnumeric() method
    if not enrolls_data.loc[idx,'SchoolRegion'].isnumeric():
      if enrolls_data.loc[idx,'SchoolRegion'] == 'I':
        enrolls_data.loc[idx,'SchoolRegion'] = 1
      elif enrolls_data.loc[idx,'SchoolRegion'] == 'II':
        enrolls_data.loc[idx,'SchoolRegion'] = 2
      elif enrolls_data.loc[idx,'SchoolRegion'] == 'III':
        enrolls_data.loc[idx,'SchoolRegion'] = 3
      elif enrolls_data.loc[idx,'SchoolRegion'] == 'IV':
        enrolls_data.loc[idx,'SchoolRegion'] = 4
      elif enrolls_data.loc[idx,'SchoolRegion'] == 'V':
        enrolls_data.loc[idx,'SchoolRegion'] = 5
      elif enrolls_data.loc[idx,'SchoolRegion'] == 'VI':
        enrolls_data.loc[idx,'SchoolRegion'] = 6
      elif enrolls_data.loc[idx,'SchoolRegion'] == 'VII':
        enrolls_data.loc[idx,'SchoolRegion'] = 7
      elif enrolls_data.loc[idx,'SchoolRegion'] == 'VIII':
        enrolls_data.loc[idx,'SchoolRegion'] = 8
      elif enrolls_data.loc[idx,'SchoolRegion'] == 'IX':
        enrolls_data.loc[idx,'SchoolRegion'] = 9
      elif enrolls_data.loc[idx,'SchoolRegion'] == 'X':
        enrolls_data.loc[idx,'SchoolRegion'] = 10
      elif enrolls_data.loc[idx,'SchoolRegion'] == 'XI':
        enrolls_data.loc[idx,'SchoolRegion'] = 11
      elif enrolls_data.loc[idx,'SchoolRegion'] == 'XII':
        enrolls_data.loc[idx,'SchoolRegion'] = 12
      elif enrolls_data.loc[idx,'SchoolRegion'] in ['XIII', 'RM']:
        enrolls_data.loc[idx,'SchoolRegion'] = 13
      elif enrolls_data.loc[idx,'SchoolRegion'] == 'XIV':
        enrolls_data.loc[idx,'SchoolRegion'] = 14
      elif enrolls_data.loc[idx,'SchoolRegion'] == 'XV':
        enrolls_data.loc[idx,'SchoolRegion'] = 15
      else:
        enrolls_data.loc[idx,'SchoolRegion'] = enrolls_data.loc[idx,'SchoolRegion'] + "_UNK"
  # EdTypeCode
  if enrolls_data.loc[idx,'EdTypeCode'] == 'H1': # Humanista Científico Diurno
    enrolls_data.loc[idx,'EdTypeCode'] = 1
  elif enrolls_data.loc[idx,'EdTypeCode'] == 'H2': # Humanista Científico Nocturno
    enrolls_data.loc[idx,'EdTypeCode'] = 2
  elif enrolls_data.loc[idx,'EdTypeCode'] == 'T1': # Técnico Profesional Comercial
    enrolls_data.loc[idx,'EdTypeCode'] = 3
  elif enrolls_data.loc[idx,'EdTypeCode'] == 'T2': # Técnico Profesional Industrial
    enrolls_data.loc[idx,'EdTypeCode'] = 4
  elif enrolls_data.loc[idx,'EdTypeCode'] == 'T3': # Técnico Profesional Servicios y Técnica
    enrolls_data.loc[idx,'EdTypeCode'] = 5
  elif enrolls_data.loc[idx,'EdTypeCode'] == 'CEFT': # Centro de Formacion Tecnica
    enrolls_data.loc[idx,'EdTypeCode'] = 6
  else:
    enrolls_data.loc[idx,'EdTypeCode'] = 0
  # SchoolType
  if enrolls_data.loc[idx,'SchoolType'] == 'Municipal':
    enrolls_data.loc[idx,'SchoolType'] = 1
  elif enrolls_data.loc[idx,'SchoolType'] == 'Particular Subvencionado':
    enrolls_data.loc[idx,'SchoolType'] = 2
  elif enrolls_data.loc[idx,'SchoolType'] in ['Particular no subvencionado', 'Particular NO Subvencionado']:
    enrolls_data.loc[idx,'SchoolType'] = 3
  elif 'Delegada' in enrolls_data.loc[idx,'SchoolType']:
    enrolls_data.loc[idx,'SchoolType'] = 4 # Corporacion Administracion Delegada
  else:
    enrolls_data.loc[idx,'SchoolType'] = 5 # Corporacion Municipal

# Grades: Performance
"""
Performance: from Grade, SubjHomologate, SubjValidate
"""
for idx, row in grades_data.iterrows():
  # Performance
  if grades_data.loc[idx,'Grade'] not in ['A', 'NA'] and not (pd.isnull(row['Grade'])):
    if 0.0 <= float(grades_data.loc[idx,'Grade']) <= 7.1 and grades_data.loc[idx,'SubjStatus'] != 4 and grades_data.loc[idx,'SubjHomologate'] != 'X':
      grades_data.loc[idx,'Performance'] = grades_data.loc[idx,'Grade']
    elif grades_data.loc[idx,'SubjStatus'] == 4:
      grades_data.loc[idx,'Performance'] = 91 #'DRP'
    elif grades_data.loc[idx,'SubjStatus'] == 3 and grades_data.loc[idx,'SubjValidate'] != 'X' and grades_data.loc[idx,'SubjHomologate'] != 'X':
      grades_data.loc[idx,'Performance'] = 92 #'BAD'
    elif grades_data.loc[idx,'SubjStatus'] == 2 and (grades_data.loc[idx,'SubjValidate'] == 'X' or grades_data.loc[idx,'SubjHomologate'] == 'X' ):
      grades_data.loc[idx,'Performance'] = 93 #'CVL'
    elif grades_data.loc[idx,'SubjStatus'] == 1:
      grades_data.loc[idx,'Performance'] = 94 #'NULL'
    else:
      grades_data.loc[idx,'Performance'] = 99 #'UNK'
  elif grades_data.loc[idx,'SubjStatus'] == 4:
    grades_data.loc[idx,'Performance'] = 91 #'DRP'
  elif grades_data.loc[idx,'SubjStatus'] == 3:
    grades_data.loc[idx,'Performance'] = 92 #'BAD'
  elif grades_data.loc[idx,'SubjStatus'] == 2:
    grades_data.loc[idx,'Performance'] = 93 #'CVL'
  elif grades_data.loc[idx,'SubjStatus'] == 1:
    grades_data.loc[idx,'Performance'] = 94 #'NULL'
  else:
    grades_data.loc[idx,'Performance'] = 99 #'UNK'

# Drop source columns after Feature Engineering
grades_drop_cols = ['SubjValidate','Grade','SubjHomologate']
grades_data.drop(grades_drop_cols, axis=1, inplace = True)
enrolls_drop_cols = ['Campus','PostulationType']
enrolls_data.drop(enrolls_drop_cols, axis=1, inplace = True)

print("-"*25 + "\nFeature Engineering Done\n" + "-"*25 )

# After Fills Cols
if stats:
  # info
  enrolls_data.info()
  print("-"*10)
  grades_data.info()
  print("-"*10)
  # describe
  enrolls_data.describe(include = 'all').to_csv("reports/FILTERS_enrolls_desc_End.csv")
  grades_data.describe(include = 'all').to_csv("reports/FILTERS_grades_desc_End.csv")
  # nulls
  print('enrolls_data columns with null values:\n', enrolls_data.isnull().sum())
  print("-"*10)
  print('grades_data columns with null values:\n', grades_data.isnull().sum())
  print("-"*10)

# Create Filtered bin CSV Files -  to use with ML models
grades_data_bin_cols = [
  'Rut','Year','Period','SubjStatus','Performance']
grades_data = grades_data[grades_data_bin_cols]
grades_data.to_csv("data/FILTERS_Grades_bin.csv", index=False)

enrolls_data_bin_cols = [ 
  'Rut','EntryYear','NEMScr','Ranking','LangScr','MathScr','ScienScr',
  'SchoolRegion','EdTypeCode','SchoolType','MotherEd','CampusStgo',
  'PostulationRegular','Desertor']
enrolls_data = enrolls_data[enrolls_data_bin_cols]
enrolls_data.to_csv("data/FILTERS_Enrolls_bin.csv", index=False)

timeEnd = time.time()
dtEnd = datetime.fromtimestamp(timeEnd)
print('-'*25 + "\nFILTERS Script End:" + str(dtEnd) + '\n' + '-'*25 + "\nTotal Time: " + str(timeEnd - timeStart) + "\n")
fTemp.close()