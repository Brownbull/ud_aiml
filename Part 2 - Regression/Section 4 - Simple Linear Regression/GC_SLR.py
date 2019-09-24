# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 

@author: Brownbull
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

# Select Features
X = dataset.iloc[:, :-1].values
# Select Target 
y = dataset.iloc[:, 1].values
 
# Split Data
from sklearn.model_selection import train_test_split
# random split 
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0) 
# fixed split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 1/3, random_state = 0) 

"""
# Feature Scaling - Put everything on the same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(train_X)
"""

# Fitting SLR to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_X, train_y)

# Predict
pred_y = regressor.predict(test_X)

# Evaluate
from sklearn.metrics import mean_absolute_error
mean_absolute_error(test_y, pred_y) 

# Visualising 
plt.scatter(train_X, train_y, color='red')
plt.plot(train_X, regressor.predict(train_X), color='blue')
plt.title("Salary vs Experience SLR prediction (training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualising 
plt.scatter(test_X, test_y, color='red')
plt.plot(train_X, regressor.predict(train_X), color='blue')
plt.title("Salary vs Experience SLR prediction (test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()









