"""
This script defines functions for data preprocessing and initialization for the machine learning model.
The primary purpose is to read an ASCII file into a DataFrame, filter and process the data to generate features,
and split the data into training and testing sets using stratified shuffle split.

Functions:
- init: Initializes the data processing pipeline, reads the file, processes the data, and splits it into training and testing sets.
- ascii_to_df: Reads an ASCII file into a DataFrame and handles potential parsing errors.
- colours: Processes the DataFrame to create color indices from flux values and prepares the feature matrix and target vector.

Input Parameters:
- seed (int): Random seed for reproducibility.
- file (str): Path to the ASCII file to be read.
- split (float): Fraction of the data to be used as the test set.
- bins (int): Number of bins to use for stratification.
- filter_order (list): List of column names representing different filters from shortest to longest effective wavelength.
- max_limit (float): Maximum redshift value to consider (default is 7).

Outputs:
- X (DataFrame): Feature matrix containing the color indices.
- y (Series): Target vector containing the redshift values.
- train_idx (array): Indices of the training set.
- test_idx (array): Indices of the testing set.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def init(seed, file, split, bins, filter_order, max_limit=7):
    # Read ASCII file into a DataFrame
    df = ascii_to_df(file)
    # Filter rows with valid redshift values and generate colour indices from flux values
    df_zspec = df[(df['zspec'] != -99.0) & (df['zspec'] <= max_limit)]
    X, y = colours(filter_order, df_zspec)
    # Initialize stratified shuffle split and split the data into training and testing sets
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=split, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, pd.qcut(y, bins, labels=False)))
    return X, y, train_idx, test_idx

# code source: RBeg
def ascii_to_df(file,**kwargs):
    try:
        df = pd.read_csv(file,sep="\\s+", low_memory=False,**kwargs)
    except pd.errors.ParserError:
        raise UserWarning('Error parsing file to DataFrame')
    if df.columns[0]=='#':
        df_cols = df.columns
        df.drop(labels=df_cols[-1],axis=1,inplace=True)
        df.columns = df_cols[1:]
    return df

def colours(filter_order, df):
    column_names = df.columns.tolist()
    df = df.copy()
    # Ensure all columns in filter_order exist in the DataFrame
    fluxes = [f for f in filter_order if f in column_names]
    if len(fluxes) != len(filter_order):
        raise ValueError("Some columns in filter_order do not exist in the DataFrame")
    # Filter out rows with invalid flux values
    mask = df[filter_order].eq(-99.0).any(axis=1)
    df = df[~mask]
    fluxes = [f for f in filter_order if f in column_names]
    # Convert flux values to magnitudes and calculate colour indices
    for flux in fluxes:
        df.loc[df[flux] <= 0, flux] = np.nan
        df[flux] = -2.5 * np.log10(df[flux])
    for i, flux1 in enumerate(fluxes[:-1]):
        flux2 = fluxes[i+1]
        colour_name = f'c_{flux2}/{flux1}'
        df.loc[:, colour_name] = df[flux2] - df[flux1]
        df.loc[:, colour_name] = df[colour_name].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    # Extract the target vector and feature matrix
    y = df['zspec']
    colours = [col for col in df.columns if col.startswith('c_')]
    X = df[colours]
    return X, y
