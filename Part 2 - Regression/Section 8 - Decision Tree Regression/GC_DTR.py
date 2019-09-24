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
from sklearn.tree import DecisionTreeRegressor 
dtr = DecisionTreeRegressor(random_state=0) 
dtr.fit(X, y)

# predict
pred_y=dtr.predict([[6.5]])
print(pred_y)

# HD Visualization
grid_X = np.arange(min(X), max(X), 0.01) # create base rangue with small intervals
grid_X = grid_X.reshape((len(grid_X), 1)) # transform to matrix
plt.scatter(X, y, color='red')
plt.plot(grid_X, dtr.predict(grid_X), color='blue')
plt.title('Truth or Bluff - DTR')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

# Evaluate
from sklearn.metrics import mean_absolute_error
predict_y = dtr.predict(X) 
mae = mean_absolute_error(y, predict_y) 
print(mae)
