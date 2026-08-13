# Generate a python script for data preprocessing
import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data.
    """
    return pd.read_csv(file_path)

def clean_data(df):
    """
    Clean the data by removing missing values and duplicates.

    Parameters:
    df (pd.DataFrame): The data to clean.

    Returns:
    pd.DataFrame: The cleaned data.
    """
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def transform_data(df):
    """
    Transform the data by encoding categorical variables and scaling numerical variables.

    Parameters:
    df (pd.DataFrame): The data to transform.

    Returns:
    pd.DataFrame: The transformed data.
    """
    # Example transformation steps (replace with actual logic)
    df = pd.get_dummies(df, columns=['category'])
    return df