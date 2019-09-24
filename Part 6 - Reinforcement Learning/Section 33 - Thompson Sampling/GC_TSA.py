# Thomson Sampling Algorithm

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implement TSA from scratch
# Number of observations
import random
N = 10000
# Number of Items to evaluate 
d = 10
# Vector of all selected in each round
ItemsSelected = []

# Initilaization
n_rewards_1 = [0] * d
n_rewards_0 = [0] * d
TotalReward = 0

# Calculation
for n in range(0, N):
    max_random = 0
    Item = 0
    for i in range(0, d):
        random_beta = random.betavariate(n_rewards_1[i] + 1, n_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            Item = i
    ItemsSelected.append(Item)
    # Link with Dataset
    reward = dataset.values[n, Item]
    if reward == 1:
        n_rewards_1[Item] = n_rewards_1[Item] + 1
    else:
        n_rewards_0[Item] = n_rewards_0[Item] + 1
    TotalReward = TotalReward + reward
    
# Visualize
plt.hist(ItemsSelected)
plt.title('TSA')
plt.xlabel('Items')
plt.ylabel('Selected Times')
plt.show()