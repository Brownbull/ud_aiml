# Natural Language Processing
# Bag of words Model

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Data
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting = 3)

# Clean Text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# Initialize corpus = collection of texts
corpus = []
for i in range(0,1000):
    # Replace numbers, special characters by space = token patern
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # LowerCase to all
    review = review.lower()
    # Split into words
    review = review.split()
    # Eliminate non-relevant words and Take Only root of word : Loving, Loved Love = Love
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # Join back spllited words
    review = ' '.join(review)
    corpus.append(review)

# Bag of Words Model: matrix of words counters
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # Keep the most relevant words
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 

# Evaluate - MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred) 
acc = 1 - mae

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


