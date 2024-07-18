"""
This script defines functions for implementing kernelized linear regression with L2 regularization using a Gaussian Radial Basis Function (RBF) kernel.
Key functionalities include computing the closed form solution for kernelized linear regression, calculating the RBF kernel, making predictions using the kernelized model, and computing evaluation metrics.

Functions:
- rbf_kernel: Computes the Gaussian RBF kernel between two matrices.
- closed_form_kernel: Computes the closed form solution for kernelized linear regression with L2 regularization.
- predict_kernel: Makes predictions using the kernelized model.
- calc_metrics: Computes evaluation metrics.

Input Parameters:
- X_train (np.ndarray): Training data features.
- Y_train (np.ndarray): Training data labels.
- X_test (np.ndarray): Test data features.
- lambda_factor (float): Regularization constant.
- gamma (float): Gamma parameter for the Gaussian RBF kernel.
- alpha (np.ndarray): Weights in the kernelized space from the kernelized model.

Outputs:
- alpha (np.ndarray): Weights in the kernelized space from the kernelized model.
- kernel_matrix (np.ndarray): Kernel matrix computed from the RBF kernel.
- predictions (np.ndarray): Predicted values for the test data.
- mse, sigma_dz, f_outlier (float): evaluation metrics
"""

import numpy as np

def rbf_kernel(X, Y, gamma):
    X = np.asarray(X)
    Y = np.asarray(Y)
    # Compute the squared Euclidean distance between each pair of points
    sq_dist = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
    # Compute the Gaussian RBF kernel matrix
    return np.exp(-gamma * sq_dist)

def closed_form_kernel(X_train, y_train, lambda_factor, gamma):
    # Compute the kernel (Gram) matrix
    K = rbf_kernel(X_train, X_train, gamma)
    # Add regularization term to the kernel matrix
    regularization_matrix = lambda_factor * np.eye(K.shape[0])
    # Compute the closed form solution for alpha
    alpha = np.linalg.inv(K + regularization_matrix).dot(y_train)
    return alpha

def predict_kernel(X_train, X_test, alpha, gamma):
    # Compute the kernel matrix between training and test data
    K_test = rbf_kernel(X_test, X_train, gamma)
    # Predict using the learned alpha weights
    predictions = K_test.dot(alpha)
    return predictions

def calc_metrics(z_phot, z_spec):
    mse = np.mean((z_phot - z_spec)**2)
    dz = (z_phot - z_spec) / (1 + z_spec)
    MAD = np.median(np.abs(dz - np.median(dz)))
    sigma_dz = 1.4826 * MAD
    outliers = np.abs(dz) > 0.15
    f_outlier = np.sum(outliers) / len(dz)
    return mse, sigma_dz, f_outlier
