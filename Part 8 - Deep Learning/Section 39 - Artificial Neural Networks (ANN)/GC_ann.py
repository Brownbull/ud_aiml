# Artificial Neural Network
# !pip install wheelsOfTheanoTensorFlowAndKeras

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
# Change countries per 0 1 2
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
# Change Gender per 0 1
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Convert countries flag to multiple binary data columns - Dummy variables
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Avoid dummy variable trap - take all column, except first one
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialize ANN
# Initialize as a sequence of layers
classifier = Sequential()

# Average of numbers of nodes on input and output nodes, (11 + 1 )/ 2 = 6
# Add Input Layer and Add First Hidden Layer using Recetifier activation Function
classifier.add(Dense(activation = 'relu', input_dim = 11, units = 6, kernel_initializer= 'uniform'))

# Add Second Hidden Layer using relu activation function
classifier.add(Dense(activation = 'relu', units = 6, kernel_initializer= 'uniform'))

# Add Output Layer using Sigmoid activation function
# TIP: in case of dependent variable have more than 1 category use kernel_initializer='softmax'
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer= 'uniform'))

# Compiling the ANN - 
# Apply Stochastic Gradient Descent Algorithm = Adam
# Loss function is sum of distances between predicted and average = 
# if binary out=> binary_crossentropy / else categorycall_crossentropy
# Metrics, Criteria to evaluate the model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit ANN to training set
# batch_size = numbers of observations sent to train de ANN
# nb_epoch = numbers of times the ANN is calibrated with batch files
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Transform results to true or false
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy Values
# Defs
TP = cm[0][0]
TN = cm[1][1]
FP = cm[0][1]
FN = cm[1][0]
# Formulas
Accuracy = (TP + TN) / (TP + TN + FP + FN) # 70 80 90 Good
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2 * Precision * Recall / (Precision + Recall) 
print('Accuracy: ', Accuracy)
print('Precision: ', Precision)
print('Recall: ', Recall)
print('F1_Score: ', F1_Score)

# Evaluate - MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred) 
acc = 1 - mae
print("MAE: ", mae)
print("ACC: ", acc)
