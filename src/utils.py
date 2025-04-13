import os
import pandas as pd
import joblib

def get_categorical_columns(df):
    """
    Get a list of categorical columns from a dataframe
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def get_numerical_columns(df):
    """
    Get a list of numerical columns from a dataframe
    """
    return df.select_dtypes(include=['int', 'float']).columns.tolist()

def check_missing_values(df):
    """
    Check for missing values in the dataframe
    """
    return df.isnull().sum()

def ensure_directory_exists(directory):
    """
    Create a directory if it doesn't exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)