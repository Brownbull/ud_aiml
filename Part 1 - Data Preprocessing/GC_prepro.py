# -*- coding: utf-8 -*-
"""
@author: Brownbull
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Select Features
X = dataset.iloc[:, : -1].values
# Select Target 
y = dataset.iloc[:, 3].values

# Inspect Columns
desc = dataset.describe()
nancheck = dataset.columns[dataset.isna().any()].tolist()

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encode Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
# transform words to numbers
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0]) 
y = labelEncoder_X.fit_transform(y) 
# Create dummy variables
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()
 
# Split Data
from sklearn.model_selection import train_test_split
# random split 
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0) 
# fixed split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# Feature Scaling - Put everything on the same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(train_X)








