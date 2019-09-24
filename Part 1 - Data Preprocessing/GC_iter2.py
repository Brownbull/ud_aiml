# -*- coding: utf-8 -*-
"""
  Preprocessing script
"""

# Importing libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import input dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Missing Data processing
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = "mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encode Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
# Apply on categorical variable
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0]) # transform words to numbers
# Transform encoded variable to different columns (dummy variables)
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()

# Split Dataset into training and test sets 
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2, random_state= 0) 

# Feature Scaling - Put everything on the same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(train_X)