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

# Fitting DTR
from sklearn.ensemble import RandomForestRegressor 
rfr = RandomForestRegressor(n_estimators = 300, random_state=0) 
rfr.fit(X, y)

# predict
pred_y=rfr.predict([[6.5]])
print(pred_y)

# HD Visualization
grid_X = np.arange(min(X), max(X), 0.001) # create base rangue with small intervals
grid_X = grid_X.reshape((len(grid_X), 1)) # transform to matrix
plt.scatter(X, y, color='red')
plt.plot(grid_X, rfr.predict(grid_X), color='blue')
plt.title('Truth or Bluff - RFR')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

# Evaluate
from sklearn.metrics import mean_absolute_error
predict_y = rfr.predict(X) 
mae = mean_absolute_error(y, predict_y) 
print(mae)
