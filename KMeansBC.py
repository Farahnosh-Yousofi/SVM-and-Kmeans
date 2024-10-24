import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# KMeans Clustering Class
class KMeansClustering:
    def __init__(self, k=2):  # Setting k=2 for 2 clusters (malignant and benign)
        self.k = k  # number of clusters
        self.centroids = None  # centroids of clusters
    
    # calculate euclidean distance
    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))
    
    # fit method
    def fit(self, X, max_iterations=200):
        # Initialize centroids randomly within the range of X
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

        for _ in range(max_iterations):
            y = []
            
            # Assign each data point to the nearest centroid
            for data_point in X:
                distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)  # Find the closest centroid
                y.append(cluster_num)
            
            y = np.array(y)  # Cluster assignments for all points
             
            # Update the centroids by calculating the mean of the assigned points
            new_centroids = []
            for i in range(self.k):
                cluster_points = X[y == i]
                if len(cluster_points) == 0:
                    new_centroids.append(self.centroids[i])  # If no points in the cluster, keep the old centroid
                else:
                    new_centroids.append(np.mean(cluster_points, axis=0))  # Recompute centroid as the mean
            
            new_centroids = np.array(new_centroids)
            
            # Check for convergence (if centroids do not change significantly)
            if np.max(np.linalg.norm(self.centroids - new_centroids, axis=1)) < 0.0001:
                break  # Stop if centroids converge
            else:
                self.centroids = new_centroids
        
        return y  # Return final cluster assignments

# Step 1: Load and prepare the breast cancer dataset
breastCancer_data = pd.read_csv('wdbc.data')
breastCancer_data.columns = ['ID', 'Diagnosis', 'radius', 'texture1', 'perimeter1', 'area1', 'smoothness1',
                             'compactness1', 'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1', 
                             'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 
                             'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2', 'radius3', 
                             'texture3', 'perimeter3', 'area3', 'smoothness3','compactness3', 'concavity3', 
                             'concave_points3', 'symmetry3', 'fractal_dimension3']

# Convert M (malignant) to 1 and B (benign) to 0 (for evaluation purposes)
breastCancer_data['Diagnosis'] = breastCancer_data['Diagnosis'].replace({'M': 1, 'B': 0})

# Remove the 'ID' column and the 'Diagnosis' column (since we don't know the labels during clustering)
X = breastCancer_data.drop(columns=['ID', 'Diagnosis']).values
y_true = breastCancer_data['Diagnosis'].values  # True labels for evaluation after clustering

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply K-Means Clustering
kmeans = KMeansClustering(k=2)
clusters = kmeans.fit(X_scaled)

# Step 3: Adjust the cluster labels to match the actual labels (based on majority class)
# We assume cluster 0 is benign (0) and cluster 1 is malignant (1).
# Flip the labels if accuracy is lower than 50%
if accuracy_score(y_true, clusters) < 0.5:
    clusters = 1 - clusters  # Switch 0 to 1 and 1 to 0

# Step 4: Evaluate the clustering performance using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_true, clusters)
precision = precision_score(y_true, clusters)
recall = recall_score(y_true, clusters)
f1 = f1_score(y_true, clusters)

# Step 5: Print the results
print(f"K-Means Clustering (from Scratch) - Accuracy: {accuracy:.2f}")
print(f"K-Means Clustering (from Scratch) - Precision: {precision:.2f}")
print(f"K-Means Clustering (from Scratch) - Recall: {recall:.2f}")
print(f"K-Means Clustering (from Scratch) - F1-Score: {f1:.2f}")