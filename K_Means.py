import numpy as np
import matplotlib.pyplot as plt

# creating class KmeansClassifier
class KMeansClustering:
    def __init__(self, k=3):
        self.k = k  # number of clusters
        self.centroids = None  # centroids of clusters
    
    # calculate euclidean distance
    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))
    
    # fit method
    def fit(self, X, max_iterations=200):
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0),
                                           size=(self.k, X.shape[1]))  # initialize centroids randomly
        for _ in range(max_iterations):
            y = []
            
            for data_point in X:
                distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)
            
            y = np.array(y)
             
            cluster_centers = []
            for i in range(self.k):
                indices = np.argwhere(y == i).flatten()
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0))
            
            cluster_centers = np.array(cluster_centers)
            
            if np.max(np.linalg.norm(self.centroids - cluster_centers, axis=1)) < 0.0001:
                break
            else:
                self.centroids = cluster_centers
        
        return y

# Testing the KMeansClustering class
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

# Generate random data
data = make_blobs(n_samples=100, n_features=2, centers=3)
random_points = data[0]

kmeans = KMeansClustering(k=3)
labels = kmeans.fit(random_points)

# Print the cluster labels
print(data[1])  # True labels
print(labels)  # Predicted labels

# Calculate the Adjusted Rand Index (ARI)
#ari = adjusted_rand_score(data[1], labels)
#print(f"Adjusted Rand Index: {ari}")

# Plot the data and centroids
plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], marker='*', s=200, c=range(len(kmeans.centroids)))
plt.show()