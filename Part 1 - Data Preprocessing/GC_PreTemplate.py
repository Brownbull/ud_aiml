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
 
# Split Data
from sklearn.model_selection import train_test_split
# random split 
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0) 
# fixed split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0) 

"""
# Feature Scaling - Put everything on the same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(train_X)
"""
