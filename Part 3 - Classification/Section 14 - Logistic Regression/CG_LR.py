# -*- coding: utf-8 -*-
"""
@author: Brownbull
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Select Features
X = dataset.iloc[:, [2,3]].values
# Select Target 
y = dataset.iloc[:, 4].values
 
# Split Data
from sklearn.model_selection import train_test_split
# random split 
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0) 
# fixed split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.25, random_state = 0) 

# Feature Scaling - Put everything on the same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)

# Fit Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(train_X, train_y)

# Predict
pred_y = classifier.predict(test_X)

# Evaluate - Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y, pred_y)
"""
Matrix will tell by variable Correct preds and incorrect
"""
# Visualising Clasiffication
from matplotlib.colors import ListedColormap
set_X, set_y = train_X, train_y
# Set All pixels positions in a matrix
X1, X2 = np.meshgrid(np.arange(start = set_X[:, 0].min() - 1, stop = set_X[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = set_X[:, 1].min() - 1, stop = set_X[:, 1].max() + 1, step = 0.01 ))
# Predict result on each pixel value and make a contour in graph(line accross)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('brown', 'green')))
# Set limits on visualization
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# plot the real values
for i, j in enumerate(np.unique(set_y)):
    plt.scatter(set_X[set_y == j, 0], set_X[set_y == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Logistic Regression (Training Set)")
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()

# Visualising Clasiffication
from matplotlib.colors import ListedColormap
set_X, set_y = test_X, test_y
X1, X2 = np.meshgrid(np.arange(start = set_X[:, 0].min() - 1, stop = set_X[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = set_X[:, 1].min() - 1, stop = set_X[:, 1].max() + 1, step = 0.01 ))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('brown', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(set_y)):
    plt.scatter(set_X[set_y == j, 0], set_X[set_y == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Logistic Regression (Test Set)")
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()


# Evaluate - MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test_y, pred_y) 







