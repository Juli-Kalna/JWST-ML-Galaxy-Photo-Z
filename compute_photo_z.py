"""
This script defines a function to compute photometric redshifts using kernelized regression with nearest neighbors.
The main function `compute_phot_z` performs the following steps:
- Identifies the nearest neighbors for each test point.
- Computes kernel regression weights using the nearest neighbors.
- Predicts photometric redshifts and estimates their errors.
- Flags test points that are extrapolated beyond the training data range.
- Refines the nearest neighbors based on residuals and re-computes predictions and errors.

Functions:
- compute_phot_z: Computes photometric redshifts for test data using kernelized regression.

Input Parameters:
- X_train (DataFrame): Training data features.
- X_test (DataFrame): Test data features.
- y_train (Series): Training data labels.
- params (dict): Dictionary of hyperparameters (lambda_factor, gamma, Nneigh).

Outputs:
- z_phot (ndarray): Predicted photometric redshifts for the test data.
- z_phot_err (ndarray): Estimated errors of the photometric redshifts.
- flag_extrapolation (ndarray): Flags indicating if the test data points are extrapolated beyond the training data range.
"""

import numpy as np
import pandas as pd
from linear_regression_kernel import closed_form_kernel, predict_kernel

def compute_phot_z(X_train, X_test, y_train, params):
    # Extract hyperparameters from params
    lambda_factor = params['lambda_factor']
    gamma = params['gamma']
    Nneigh = int(params['Nneigh'])

    Ntest = X_test.shape[0]
    flag_extrapolation = np.zeros(Ntest, dtype='float')
    z_phot = np.zeros(Ntest, dtype='float')
    z_phot_err = np.zeros(Ntest, dtype='float')
    # Iterate over each test point
    for i in range(Ntest):
        # find the Nneigh nearest neighbors
        distances = np.linalg.norm(X_train - X_test.iloc[i, :], axis=1)
        Nneigh_idx = np.argpartition(distances, Nneigh)[:Nneigh]
        X_train_neigh = X_train.iloc[Nneigh_idx, :]
        y_train_neigh = y_train.iloc[Nneigh_idx]

        # Compute kernel regression weights using nearest neighbors
        alpha = closed_form_kernel(X_train_neigh, y_train_neigh, lambda_factor, gamma)
        # Predict the photometric redshift for the test point
        z_phot[i] = predict_kernel(X_train_neigh, X_test.iloc[i:i+1, :], alpha, gamma)[0]
        if z_phot[i] < 0:
            z_phot[i] = 0

        # Estimate the prediction error
        z_phot_neigh = predict_kernel(X_train_neigh, X_train_neigh, alpha, gamma)
        residuals = y_train_neigh - z_phot_neigh
        z_phot_err[i] = np.mean(np.abs(residuals))
        # Flag if the test point is an extrapolation
        if np.any(np.logical_or((X_test.iloc[i, :] < np.amin(X_train_neigh, axis=0)), (X_test.iloc[i, :] > np.amax(X_train_neigh, axis=0)))):
            flag_extrapolation[i] = 1
        # Refinement step for extrapolated points
        neigh_index = np.abs(residuals) <= 3 * z_phot_err[i]
        ng = np.count_nonzero(neigh_index)
        if ng < Nneigh and ng > 0:
            # Refine neighbors based on the new index
            X_train_neigh = X_train_neigh[neigh_index]
            y_train_neigh = y_train_neigh[neigh_index]
            alpha = closed_form_kernel(X_train_neigh, y_train_neigh, lambda_factor, gamma)
            z_phot[i] = predict_kernel(X_train_neigh, X_test.iloc[i:i+1, :], alpha, gamma)[0]
            z_phot_neigh = predict_kernel(X_train_neigh, X_train_neigh, alpha, gamma)
            residuals = y_train_neigh - z_phot_neigh
            z_phot_err[i] = np.std(residuals)  # Use standard deviation again
            # Check for extrapolation with refined neighbors
            if np.any((X_test.iloc[i, :] < X_train_neigh.min(axis=0)) | (X_test.iloc[i, :] > X_train_neigh.max(axis=0))):
                flag_extrapolation[i] = 1

    return z_phot, z_phot_err, flag_extrapolation
