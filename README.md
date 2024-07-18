# Predicting Astronomical Redshifts using Kernelized Regression
## Overview
This repository contains a machine learning pipeline designed to predict astronomical redshifts using kernelized regression. The pipeline involves data preprocessing, hyperparameter optimization using Bayesian optimization, model training, and evaluation.
## Pipeline Details
### Data Preprocessing
The data preprocessing steps are handled in the preprocessing.py script. This includes reading the ASCII file, filtering and processing the data to generate features, and splitting the data into training and testing sets using stratified shuffle split.
### Hyperparameter Optimization
Hyperparameter optimization is performed using Bayesian optimization, implemented in the bayesian_optimization.py script. The optimizer searches for the best parameters for the kernelized regression model.
### Model Training and Evaluation
The model training and evaluation steps are conducted in the compute_photo_z.py and linear_regression_kernel.py scripts. The kernelized regression model is trained using the optimal hyperparameters, and its performance is evaluated on the test set.
## Input Parameters
The main function of the pipeline accepts the following input parameters:
- file (str): Path to the ASCII file to be read.
- filter_order (list): List of column names representing different filters, ordered from shortest to longest wavelength.
- seed (int): Random seed for reproducibility.
- split (float): Fraction of the data to be used as the test set.
- bins (int): Number of bins to use for stratification.
- folds (int): Number of cross-validation folds.
- n_iter (int): Number of iterations for Bayesian optimization.
These parameters can be configured directly in the main.py script.
## Outputs
The pipeline prints the following evaluation metrics for the optimized model on the test set:
- Mean Squared Error (MSE)
- Sigma_dz
- Fraction of outliers (f_outlier)
These metrics help assess the accuracy and reliability of the predicted photometric redshifts.
