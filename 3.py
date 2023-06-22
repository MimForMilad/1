import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

# Load the dataset
heart_data = pd.read_csv('heart.csv')

# Preprocess the data
X = heart_data.drop('target', axis=1)
y = heart_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add bias term to X_train and X_test
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Initialize model parameters
theta = np.zeros(X_train.shape[1])

# Set regularization parameter
alpha = 0.01

# Define sigmoid function
def sigmoid(z):
    return expit(z)

# Define cost function
def cost_function(theta, X, y, alpha):
    m = len(y)
    h = sigmoid(X @ theta)
    J = (-1/m) * (y @ np.log(h) + (1-y) @ np.log(1-h)) + (alpha/(2*m)) * (theta[1:] @ theta[1:])
    grad = (1/m) * (X.T @ (h-y)) + (alpha/m) * np.concatenate(([0], theta[1:].reshape(-1,1)))
    return J, grad

# Define gradient descent function
def gradient_descent(theta, X, y, alpha, num_iters):
    m = len(y)
    J_history = []
    for i in range(num_iters):
        J, grad = cost_function(theta, X, y, alpha)
        J_history.append(J)
        theta = theta - alpha * grad
    return theta, J_history

# Train the model using gradient descent
theta, J_history = gradient_descent(theta, X_train, y_train, alpha, 1000)

# Predict on the test set and calculate AUC
h = sigmoid(X_test @ theta)
y_pred = (h >= 0.5).astype(int)
tp = np.sum((y_test == 1) & (y_pred == 1))
fp = np.sum((y_test == 0) & (y_pred == 1))
tn = np.sum((y_test == 0) & (y_pred == 0))
fn = np.sum((y_test == 1) & (y_pred == 0))
recall = tp / (tp + fn)
precision = tp / (tp + fp)
fpr = fp / (fp + tn)
tpr = tp / (tp + fn)
auc = (1 + recall - fpr) / 2
print('AUC:', auc)
