# -*- coding: utf-8 -*-
"""
@author: Brownbull
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

# Select Features
X = dataset.iloc[:, : -1].values
# Select Target 
y = dataset.iloc[:, 4].values

# Encode Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
# transform words to numbers
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3]) 
# Create dummy variables
oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()
 
# Avoid Dummy variable trap
X = X[:, 1:]

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

# Fitting MLR
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_X, train_y)

# Predict
pred_y = model.predict(test_X)

# Evaluate
from sklearn.metrics import mean_absolute_error
mean_absolute_error(test_y, pred_y) 

# Refine with Backward Elimination
import statsmodels.formula.api as sm
# Add column of 1s to X as b0 constant
X = np.append(arr = np.ones(shape = (50,1)), values = X.astype(int), axis = 1)

# Features to have in the model
# original: opt_X = X[:,[0,1,2,3,4,5]]
opt_X = X[:,[0,3,5]]
# Backward Elimination
# 1 - significance 0.05
# 2 - Fit model with all possible predictors
model_OLS = sm.OLS(endog = y, exog = opt_X).fit()
# 3 - consider Feature with highest p-value
model_OLS.summary()
# remove features not allowed
# remove x3 feature index 2
# remove x2 feature index 1
# remove x2 feature index 4
# remove x2 feature index 5 -- Maybe

"""
#Backward Elimination with p-values and Adjusted R Squared:
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
"""