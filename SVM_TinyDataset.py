import numpy as np

class SimpleSVM:
    def __init__(self, learning_rate=0.001, iterations=1000, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_param = lambda_param
        self.w = None
        self.b = None

    def fit(self, X, y):
        m, n = X.shape  # Number of samples (m) and features (n)
        self.w = np.zeros(n)  # Initialize weights
        self.b = 0  # Initialize bias

        # Encode labels: 1 for positive class, -1 for negative class
        y_encoded = np.where(y <= 0, -1, 1)

        # Perform gradient descent
        for _ in range(self.iterations):
            for i in range(m):
                condition = y_encoded[i] * (np.dot(X[i], self.w) - self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.w  # Regularization only
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - np.dot(X[i], y_encoded[i])  # Margin violation
                    db = -y_encoded[i]
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

    def predict(self, X):
        # Predict class labels
        y_pred = np.dot(X, self.w) - self.b
        return np.where(y_pred >= 0, 1, 0)

# Test the SVM on a small dataset
X = np.array([[2, 3], [1, 1], [2, 1], [3, 2], [4, 5], [5, 6]])
y = np.array([1, 1, 0, 0, 1, 1])  # 1 for malignant, 0 for benign

# Initialize the SVM model
svm = SimpleSVM(learning_rate=0.001, iterations=1000, lambda_param=0.01)
svm.fit(X, y)

# Predict using the trained model
predictions = svm.predict(X)
print("Predictions:", predictions)