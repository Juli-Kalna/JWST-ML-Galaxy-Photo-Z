"""
This script defines the main function for executing a machine learning pipeline to predict astronomical redshifts using kernelized regression.
It includes data preprocessing, hyperparameter optimization using Bayesian optimization, model training, and evaluation.

Input Parameters:
- file (str): Path to ASCII file to be read.
- filter_order (list): List of column names representing different filters (as in file), in order from shortest to longest wavelength.
- seed (int): Random seed for reproducibility.
- split (float): Fraction of the data to be used as the test set.
- bins (int): Number of bins to use for stratification.
- folds (int): Number of cross-validation folds.
- n_iter (int): Number of iterations for Bayesian optimization.

Outputs:
- Prints the evaluation metrics (MSE, sigma_dz, f_outlier) for the optimized model evaluated on test set of the data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from preprocessing import init
from bayesian_optimization import optimize_model
from compute_photo_z import compute_phot_z
from linear_regression_kernel import calc_metrics

def main():
    # Initialize parameters
    file = 'all_with_z_spec.ascii'
    filter_order = ['f606w', 'f814w', 'f090w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']
    seed = 0
    split = 0.2
    bins = 14
    folds = 5
    n_iter = 50

    # Data initialization and splitting
    features, z_spec, train_idx, test_idx = init(seed, file, split, bins, filter_order)
    X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = z_spec.iloc[train_idx], z_spec.iloc[test_idx]
    Ntrain = X_train.shape[0]
    Ntest = X_test.shape[0]

    # Standardize the data using the entire training set
    mu_X = X_train.mean(axis=0)
    sigma_X = X_train.std(axis=0)
    X_train_norm = np.divide((X_train - np.array([mu_X,]*Ntrain)) , np.array([sigma_X,]*Ntrain))
    X_test_norm = np.divide((X_test - np.array([mu_X,]*Ntest)) , np.array([sigma_X,]*Ntest))

    # Define folds for cross-validation
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_idx = list(skf.split(X_train_norm, pd.qcut(y_train, bins, labels=False)))
    # Hyperparameter optimization
    best_params = optimize_model(seed, X_train_norm, y_train, fold_idx, n_iter)
    print(f"Best parameters: {best_params}")

    # Model prediction and evaluation
    z_phot, z_phot_err, flag_extrapolation = compute_phot_z(X_train_norm, X_test_norm, y_train, best_params)
    MSE, sigma_dz, f_outlier = calc_metrics(z_phot, y_test)
    print('Linear Regression MSE =', MSE)
    print('Linear Regression sigma dz =', sigma_dz)
    print('Linear Regression f_outlier =', f_outlier)
    return z_phot, z_phot_err, flag_extrapolation

if __name__ == "__main__":
    main()
