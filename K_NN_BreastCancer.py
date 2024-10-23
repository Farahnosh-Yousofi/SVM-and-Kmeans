import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Step 1: Load the breast cancer dataset
breastCancer_data = pd.read_csv('wdbc.data')
breastCancer_data.columns = ['ID', 'Diagnosis', 'radius', 'texture1', 'perimeter1', 'area1', 'smoothness1',
                             'compactness1', 'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1', 
                             'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 
                             'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2', 'radius3', 
                             'texture3', 'perimeter3', 'area3', 'smoothness3','compactness3', 'concavity3', 
                             'concave_points3', 'symmetry3', 'fractal_dimension3']

# Convert M (malignant) to 1 and B (benign) to 0
breastCancer_data['Diagnosis'] = breastCancer_data['Diagnosis'].replace({'M': 1, 'B': 0})

# Remove the 'ID' column
breastCancer_data = breastCancer_data.drop(columns=['ID'])

# Step 2: Split the data into features (X) and target (y)
X = breastCancer_data.drop(columns=['Diagnosis'])
y = breastCancer_data['Diagnosis']

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# K-NN Functions
def calculate_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))

def get_neighbors(train, test_row, num_neighbors):
    distances = [(index, calculate_distance(test_row, train_row)) for index, train_row in enumerate(train)]
    distances.sort(key=lambda tup: tup[1])
    neighbors_indices = [distances[i][0] for i in range(num_neighbors)]
    return neighbors_indices

def predict_classification(train, train_labels, test_row, num_neighbors):
    neighbor_indices = get_neighbors(train, test_row, num_neighbors)
    neighbor_labels = train_labels.values[neighbor_indices]  # Convert train_labels to NumPy array and index it
    prediction = np.bincount(neighbor_labels).argmax()  # Predict the most common label
    return prediction

def find_best_k(X_train, y_train, X_val, y_val):
    k_values = range(1, 21)
    best_k = 1
    best_score = 0
    for k in k_values:
        predictions = [predict_classification(X_train, y_train, row, k) for row in X_val]
        accuracy = accuracy_score(y_val, predictions)
        if accuracy > best_score:
            best_score = accuracy
            best_k = k
    return best_k, best_score

# Find the best k
best_k, _ = find_best_k(X_train, y_train, X_val, y_val)
print(f"Best k: {best_k}")

# Evaluate on the test set
predictions = [predict_classification(X_train, y_train, row, best_k) for row in X_test]
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"k-NN - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")