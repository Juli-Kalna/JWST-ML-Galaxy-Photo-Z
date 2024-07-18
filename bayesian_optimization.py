"""
This script defines functions for cross-validation and hyperparameter optimization of a model using Bayesian Optimization.
Key functionalities include model evaluation with cross-validation and hyperparameter tuning to minimize the mean squared error (MSE).

Functions:
- model_cv: Performs cross-validation for a given set of hyperparameters.
- optimize_model: Uses Bayesian Optimization to find the best hyperparameters for the model.
The model_cv function is called from within the optimize_model function to evaluate the model performance for a given set of hyperparameters.

Input Parameters:
- params (dict): Dictionary of hyperparameters (lambda_factor, gamma, Nneigh).
- X_train (DataFrame): Training data features.
- y_train (Series): Training data labels.
- fold_ind (list): List of tuples with training and validation indices for cross-validation.
- seed (int): Random seed for reproducibility.
- n_iter (int): Number of iterations for the Bayesian optimizer.

Outputs:
- val_metrics (list): List of MSE values for each cross-validation fold.
- optimizer.max['params'] (dict): Best hyperparameters found by Bayesian Optimization.
"""

import numpy as np
from bayes_opt import BayesianOptimization
from compute_photo_z import compute_phot_z

def model_cv(params, X_train, y_train, fold_idx):

    val_metrics = []
    # Loop over each fold's train and validation indices
    for train_idx, val_idx in fold_idx:
        # Extract training and validation sets
        X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]
        # Compute predictions and calculate MSE for current fold
        z_phot, _, _ = compute_phot_z(X_train_fold, X_val_fold, y_train_fold, params)
        mse = np.mean((z_phot - y_val_fold)**2)
        val_metrics.append(mse)
    # Return the negative mean of the validation metrics to be minimized
    return -np.mean(val_metrics)

def optimize_model(seed, X_train, y_train, fold_ind, n_iter):
    # Define the objective function for the optimizer
    def kernel_crossval(lambda_factor, gamma, Nneigh):
        params = {
            'lambda_factor': lambda_factor,
            'gamma': gamma,
            'Nneigh' : int(Nneigh)
        }
        # Evaluate the model with the current set of hyperparameters
        val_metrics = model_cv(params, X_train, y_train, fold_ind)
        return val_metrics
    # Initialize the Bayesian optimizer
    optimizer = BayesianOptimization(
        f = kernel_crossval,
        pbounds={
            'lambda_factor': (1e-3, 1e0),
            'gamma': (1e-3, 1e-1),
            'Nneigh': (20, 100),
        },
        random_state=seed,
        verbose=2
    )
    # Run the optimization process for the specified number of iterations
    optimizer.maximize(n_iter=n_iter)
    # Return the best hyperparameters found by the optimizer
    return optimizer.max['params']
