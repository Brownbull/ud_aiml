# KNN Means
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using elbow method to find optimal number of custers
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(
            n_clusters = i, 
            init = 'k-means++', 
            max_iter = 300, 
            n_init = 10, 
            random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elboew Method')
plt.xlabel('Number of CLusters')
plt.ylabel('WCSS')
plt.show()

# Applying k-means to dataset
kmeans = KMeans(
            n_clusters = 5, 
            init = 'k-means++', 
            max_iter = 300, 
            n_init = 10, 
            random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualize 2D only
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, color = 'red', label = 'Careful')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, color = 'blue', label = 'Standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, color = 'green', label = 'Target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, color = 'cyan', label = 'Careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, color = 'magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('KNN Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

