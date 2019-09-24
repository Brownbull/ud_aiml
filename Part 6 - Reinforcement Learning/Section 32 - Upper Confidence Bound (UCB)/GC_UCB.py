# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implement UCB from scratch
# Number of observations
from math import sqrt, log
N = 10000
# Number of Items to evaluate 
d = 10
# Vector of all selected in each round
ItemsSelected = []

# Initilaization
n_selections = [0] * d
SumRewards = [0] * d
TotalReward = 0

# Calculation
for n in range(0, N):
    MaxUpperBound = 0
    Item = 0
    for i in range(0, d):
        if n_selections[i] > 0:
            AvgReward = SumRewards[i] / n_selections[i]
            delta_i = sqrt(3/2 * log(n + 1) / n_selections[i])
            UpperBound = AvgReward + delta_i
        else:
            UpperBound = 1e400
        if UpperBound > MaxUpperBound:
            MaxUpperBound = UpperBound
            Item = i
    ItemsSelected.append(Item)
    n_selections[Item] = n_selections[Item] + 1
    # Link with Dataset
    reward = dataset.values[n, Item]
    SumRewards[Item] = SumRewards[Item] + reward
    TotalReward = TotalReward + reward
    
# Visualize
plt.hist(ItemsSelected)
plt.title('UCB')
plt.xlabel('Items')
plt.ylabel('Selected Times')
plt.show()