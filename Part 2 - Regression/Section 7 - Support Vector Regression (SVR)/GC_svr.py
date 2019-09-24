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
X = dataset.iloc[:, [1]].values
# Select Target 
y = dataset.iloc[:, [2]].values

# Feature Scaling - Put everything on the same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X, y)

# Predicting
pred_y = svr.predict(sc_X.transform([[6.5]]))
# Inverse Scaling
pred_y = sc_y.inverse_transform(pred_y)
print(pred_y)

# Visualising SVR
plt.scatter(X, y, color ='red')
plt.plot(X, svr.predict(X), color = 'blue')
plt.title('Truth or Bluff - SVR')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Evaluate
from sklearn.metrics import mean_absolute_error
predict_y = svr.predict(X) 
mae = mean_absolute_error(y, predict_y) 
print(mae)