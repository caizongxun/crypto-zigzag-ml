import os
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from typing import Dict, List, Tuple

def load_config(config_path: str = 'config.yaml') -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def time_series_split(df: pd.DataFrame, train_ratio: float = 0.8, 
                     validation_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        train_ratio: Ratio for training set
        validation_ratio: Ratio for validation set
        
    Returns:
        train_df, val_df, test_df
    """
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * validation_ratio)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    
    return train_df, val_df, test_df

def normalize_data(data: np.ndarray, method: str = 'zscore') -> Tuple[np.ndarray, Dict]:
    """
    Normalize data using specified method.
    Converts to float64 and handles edge cases.
    
    Args:
        data: Input array
        method: 'minmax' or 'zscore'
        
    Returns:
        Normalized data and normalization parameters
    """
    # Convert to float64 to ensure numeric operations
    try:
        data = np.array(data, dtype=np.float64)
    except (ValueError, TypeError) as e:
        print(f'Error converting data to float64: {e}')
        print(f'Data shape: {np.asarray(data).shape}')
        print(f'Data dtype: {np.asarray(data).dtype}')
        raise
    
    params = {}
    
    if method == 'minmax':
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        # Avoid division by zero
        range_data = data_max - data_min
        range_data = np.where(range_data == 0, 1e-8, range_data)
        normalized = (data - data_min) / (range_data + 1e-8)
        params = {'min': data_min, 'max': data_max, 'method': 'minmax'}
    
    elif method == 'zscore':
        # Calculate mean
        data_mean = np.mean(data, axis=0, dtype=np.float64)
        # Calculate standard deviation
        data_std = np.std(data, axis=0, dtype=np.float64)
        # Avoid division by zero
        data_std = np.where(data_std == 0, 1e-8, data_std)
        # Normalize
        normalized = (data - data_mean) / (data_std + 1e-8)
        params = {'mean': data_mean, 'std': data_std, 'method': 'zscore'}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.astype(np.float64), params

def denormalize_data(data: np.ndarray, params: Dict) -> np.ndarray:
    """
    Denormalize data using saved parameters.
    
    Args:
        data: Normalized array
        params: Normalization parameters
        
    Returns:
        Denormalized data
    """
    if params['method'] == 'minmax':
        data_min = params['min']
        data_max = params['max']
        return data * (data_max - data_min) + data_min
    
    elif params['method'] == 'zscore':
        data_mean = params['mean']
        data_std = params['std']
        return data * data_std + data_mean
    
    return data

def calculate_class_weights(labels: np.ndarray) -> Dict:
    """
    Calculate class weights to handle imbalanced data.
    
    Args:
        labels: Array of class labels
        
    Returns:
        Dictionary of class weights
    """
    unique_labels = np.unique(labels)
    n_samples = len(labels)
    weights = {}
    
    for label in unique_labels:
        count = np.sum(labels == label)
        weight = n_samples / (len(unique_labels) * count)
        weights[label] = weight
    
    return weights

def print_data_info(df: pd.DataFrame, name: str = 'Data'):
    """
    Print information about DataFrame.
    
    Args:
        df: Input DataFrame
        name: Name for display
    """
    print(f'\n{name} Information:')
    print(f'Shape: {df.shape}')
    print(f'Columns: {df.columns.tolist()}')
    print(f'Data types:\n{df.dtypes}')
    print(f'Missing values:\n{df.isnull().sum()}')
    print(f'\nFirst few rows:')
    print(df.head())
