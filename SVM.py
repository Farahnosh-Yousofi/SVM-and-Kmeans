#Importing numpy library
import numpy as np

# improt pandas library to convert csv to pandas format dataframe
import pandas as pd

# import StandardScaler library for standard scaling of dataframe columns
from sklearn.preprocessing import StandardScaler

# import train_test_split library for spliting dataframe into training and test sets
from sklearn.model_selection import train_test_split

# import accuracy_score library for checking accuracy of training and test sets
from sklearn.metrics import accuracy_score





'''
SVM Classifier

Equation of the Hyperplane:
y = wx - b

Gradient Descent:
Gradient Descent is an optimization algorithm used for minimizing the loss function in various machine learning algorithms. It is used for updating the parameters of the learning model.
W = w - a*dw
b=b-a*db


'''
class SVM_classifier():
    
    ## initiating the hyperparameters
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):  
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter= lambda_parameter
        
        
    # fitting the dataset to SVM Classifier  
    def fit(self, X, Y):
        # m --> Number of data points --> Number of Rows in X
        # n --> Number of features --> Number of Columns in X
        m, n = X.shape
        
        #initiating the weight and bias values
        self.w = np.zeros(n)
        self.b = 0
        
        self.X = X
        self.Y = Y
        
        # implementing gradient descent algorithm for optimization
        for i in range(self.no_of_iterations):
            self.update_weights()
        
       
      
    # updating the weights and bias  valueafter each iteration
    def update_weights(self):
        
        #label encoding: SVM works good with 1 and -1 labels so we encode 0 to -1
        y_label = np.where(self.Y <= 0, -1, 1)
        
        #fradients (dw, db)
        for index, x_i in enumerate(self.X):
            condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1
            if condition:
                dw = (2 * self.lambda_parameter * self.w)
                db = 0
            else:
                dw = (2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index]))
                db = y_label[index]
            
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
    
    
    # predicting the class labels for new data points
    def predict(self, X):
        output = np.dot(X, self.w) - self.b
        predicted_labels = np.sign(output)
        
        y_hat = np.where(predicted_labels <= -1, 0, 1)
        return y_hat
    

# Step 1: Load the dataset from csv file to pandas dataframe

breastCancer_data = pd.read_csv('wdbc.data')
breastCancer_data.columns = ['ID', 'Diagnosis', 'radius', 'texture1', 'perimeter1', 'area1', 'smoothness1','compactness1', 'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3','compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3']


# print the first 5 row of the dataframe
print(breastCancer_data.head())

# number of rows and columns in the dataframe
print(breastCancer_data.shape)

# Getting the statistical measure of the dataset
print(breastCancer_data.describe())

# getting the diagnosis
diagnosis = breastCancer_data['Diagnosis'].value_counts
print(diagnosis)