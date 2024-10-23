import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Define the Perceptron predict function
def predict(row, weights):
    activation = np.dot(row, weights)
    return 1 if activation >= 0 else 0

# Step 2: Define the function to update weights
def update_weights(row, weights, label, learning_rate):
    weights += learning_rate * (label - predict(row, weights)) * row
    return weights

# Step 3: Train Perceptron using gradient descent
def train_perceptron(X, y, epochs, learning_rate):
    weights = np.zeros(X.shape[1])  # Initialize weights with zeros
    for epoch in range(epochs):
        for i in range(len(y)):
            weights = update_weights(X[i], weights, y[i], learning_rate)
    return weights

# Step 4: Load the breast cancer dataset
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

# Step 5: Split the data into features (X) and target (y)
X = breastCancer_data.drop(columns=['Diagnosis']).values
y = breastCancer_data['Diagnosis'].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Train the Perceptron model
learning_rate = 0.01
epochs = 1000
weights = train_perceptron(X_train, y_train, epochs, learning_rate)

# Step 8: Make predictions on the training and test data
y_train_pred = np.array([predict(X_train[i], weights) for i in range(len(X_train))])
y_test_pred = np.array([predict(X_test[i], weights) for i in range(len(X_test))])

# Step 9: Calculate the evaluation metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# Step 10: Print the results
print(f"Perceptron - Training Accuracy: {train_accuracy}")
print(f"Perceptron - Test Accuracy: {test_accuracy}")
print(f"Perceptron - Precision: {precision}")
print(f"Perceptron - Recall: {recall}")
print(f"Perceptron - F1-Score: {f1}")