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
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score





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
diagnosis = breastCancer_data['Diagnosis'].value_counts()
print(diagnosis)

# change M (Malignant) and B (Benign) to 1 and 0
breastCancer_data['Diagnosis'] = breastCancer_data['Diagnosis'].replace({'M': 1, 'B': 0})

# Delete the id column from dataframe
breastCancer_data = breastCancer_data.drop('ID', axis=1)


# move dianosis column to the end
diagnosis_column = breastCancer_data.pop('Diagnosis')
breastCancer_data['Diagnosis'] = diagnosis_column

print(breastCancer_data.head())

# Split the dataset into features (X) and target (Y)
X = breastCancer_data.drop('Diagnosis', axis=1)
Y = breastCancer_data['Diagnosis']
print(X, Y)

#Data Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)


# Split the dataset into training set validation and test set
# Step 1: Split data into training + validation and test sets (80% train + validation, 20% test)
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=2)

# Step 2: Split the training + validation set into training and validation sets (80% train, 20% validation)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.25, random_state=2)

print(X_scaled.shape, X_train.shape, X_val.shape, X_test.shape)

# Step 3: Create and train the SVM Classifier
svm_classifier = SVM_classifier(learning_rate=0.001, no_of_iterations=1000, lambda_parameter=0.01)
svm_classifier.fit(X_train, Y_train)

# accuracy on training data
y_train_pred = svm_classifier.predict(X_train)
accuracy_train = accuracy_score(Y_train, y_train_pred)
print("Training Accuracy:", accuracy_train)
# Precision, Recall, F1-score for 

# accuracy on validation data
y_val_pred = svm_classifier.predict(X_val)
# Step 5: Calculate Accuracy, Precision, Recall, and F1-Score for the validation set
print("Validation Set Metrics:\n")
print(f"Accuracy: {accuracy_score(Y_val, y_val_pred):.2f}")
print(f"Precision: {precision_score(Y_val, y_val_pred):.2f}")
print(f"Recall: {recall_score(Y_val, y_val_pred):.2f}")
print(f"F1-Score: {f1_score(Y_val, y_val_pred):.2f}")


# accuracy on test data
y_test_pred = svm_classifier.predict(X_test)
# Step 6: Calculate Accuracy, Precision, Recall, and F1-Score for the test set
print("\nTest Set Metrics:\n")
print(f"Accuracy: {accuracy_score(Y_test, y_test_pred):.2f}")
print(f"Precision: {precision_score(Y_test, y_test_pred):.2f}")
print(f"Recall: {recall_score(Y_test, y_test_pred):.2f}")
print(f"F1-Score: {f1_score(Y_test, y_test_pred):.2f}")


# Building a Predective System
input_data = (14.71,21.59,95.55,656.9,0.1137,0.1365,0.1293,0.08123,0.2027,0.06758,0.4226,1.15,2.735,40.09,0.003659,0.02855,0.02572,0.01272,0.01817,0.004108,17.87,30.7,115.7,985.5,0.1368,0.429,0.3587,0.1834,0.3698,0.1094)

# Change the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardize the input data
input_data_scaled = scaler.transform(input_data_reshaped)
print(input_data_scaled)

# predict the class label for the input data
predicted_class = svm_classifier.predict(input_data_scaled)
print("Predicted Class:", predicted_class)

if predicted_class == 1:
    print("The given input data represents a Malignant tumor.")
    print("Please consult with a healthcare professional for further diagnosis.")
else:
    print("The given input data represents a Benign tumor.")