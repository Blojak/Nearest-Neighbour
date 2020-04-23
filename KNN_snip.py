"""
Benjamin Lojak:
    A simple "nearest neighbour" snip with K = 2
"""
#  Impot numpy and others
import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()


# Create data
rand    = np.random.RandomState(42) # Seed
X       = rand.randint(100, size=(10,2))
Y       = np.random.rand(10,2)

# Plot 
plt.scatter(X[:,0],X[:,1], s =100) # s =100 size of dots

# Compute (coordinate) difference with broadcasting
differences = X[:,np.newaxis,:]-X[np.newaxis,:,:]

# square coordinates
sq_differences = differences**2

# Sum up over differences
dist_sq = sq_differences.sum(-1) # row sum (0)= col sum
dist_sq.diagonal()

# Square it
dist_sq_eineZeile = np.sum(X[:,np.newaxis, :]-X[np.newaxis,:, :]**2,axis=-1)


# Determine nearest neighbour
nearest = np.argsort(dist_sq, axis = 1)
#  sorts along the last axis

print(nearest)

K = 2 # number of neighbours
nearest_partition   = np.argpartition(dist_sq, K+1, axis=1) # returns indices
nearest_part_value  = np.partition(dist_sq, K+1, axis=1) # returns values
print(nearest_partition)

plt.scatter(X[:,0],X[:,1], s =100)
# connection lines
for i in range(X.shape[0]):
    for j in nearest_partition[i,:K+1]:
        plt.plot(*zip(X[j],X[i]),color='black')
