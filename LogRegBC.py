import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 2: Define the function to calculate the cost (loss)
def cost_function(X, y, weights):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights))
    cost = -(1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

# Step 3: Define the gradient descent function
def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []
    
    for _ in range(iterations):
        predictions = sigmoid(np.dot(X, weights))
        weights -= (learning_rate / m) * np.dot(X.T, predictions - y)
        cost = cost_function(X, y, weights)
        cost_history.append(cost)
        
    return weights, cost_history

# Step 4: Define the predict function
def predict(X, weights):
    return [1 if i >= 0.5 else 0 for i in sigmoid(np.dot(X, weights))]

# Step 5: Load the breast cancer dataset
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

# Step 6: Split the data into features (X) and target (y)
X = breastCancer_data.drop(columns=['Diagnosis']).values
y = breastCancer_data['Diagnosis'].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Add an intercept term to the dataset (for the bias term)
X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

# Step 8: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 9: Initialize weights and set hyperparameters
weights = np.zeros(X_train.shape[1])
learning_rate = 0.01
iterations = 1000

# Step 10: Train the logistic regression model using gradient descent
final_weights, cost_history = gradient_descent(X_train, y_train, weights, learning_rate, iterations)

# Step 11: Make predictions on the training and test sets
y_train_pred = predict(X_train, final_weights)
y_test_pred = predict(X_test, final_weights)

# Step 12: Calculate the evaluation metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# Step 13: Print the results
print(f"Logistic Regression - Training Accuracy: {train_accuracy}")
print(f"Logistic Regression - Test Accuracy: {test_accuracy}")
print(f"Logistic Regression - Precision: {precision}")
print(f"Logistic Regression - Recall: {recall}")
print(f"Logistic Regression - F1-Score: {f1}")