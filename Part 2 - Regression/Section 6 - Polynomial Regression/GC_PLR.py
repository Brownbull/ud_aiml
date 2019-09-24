# -*- coding: utf-8 -*-
"""
@author: Brownbull
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Select Features
X = dataset.iloc[:, [1]].values # [1] to have a matrix
# Select Target 
y = dataset.iloc[:, 2].values

#  Fitting SLR
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)

#  Fitting PLR
from sklearn.preprocessing import PolynomialFeatures
plr = PolynomialFeatures(degree = 3) # degree of  polinomial
poly_X = plr.fit_transform(X) # add polinomial factor as columns
slr_2 = LinearRegression()
slr_2.fit(poly_X, y) # Regression with polinomial stuff

# Visualize SLR results
plt.scatter(X, y, color='red')
plt.plot(X, slr.predict(X), color='blue')
plt.title('Truth or Bluff - SLR')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

# Visualize PLR results
plt.scatter(X, y, color='red')
plt.plot(X, slr_2.predict(poly_X), color='blue')
plt.title('Truth or Bluff - PLR')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

grid_X = np.arange(min(X), max(X), 0.1) # create base rangue with small intervals
grid_X = grid_X.reshape((len(grid_X), 1)) # transform to matrix
plt.scatter(X, y, color='red')
plt.plot(grid_X, slr_2.predict(plr.fit_transform(grid_X)), color='blue')
plt.title('Truth or Bluff - PLR')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

# Evaluate
from sklearn.metrics import mean_absolute_error
predict_y1 = slr.predict(X) 
mae1 = mean_absolute_error(y, predict_y1) 
predict_y2 = slr_2.predict(poly_X) 
mae2 = mean_absolute_error(y, predict_y2) 
print("MAE SLR",mae1)
print("MAE PLR",mae2)

# Predicting new result with SLR
slr.predict([[6.5]])
# Predicting new result with PLR
slr_2.predict(plr.fit_transform([[6.5]]))