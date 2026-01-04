import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.zigzag_indicator import ZigZagIndicator
from src.features import FeatureEngineer

def test_zigzag_basic():
    """
    Test basic ZigZag functionality.
    """
    print('Test 1: ZigZag basic functionality')
    
    # Create sample data
    n = 100
    np.random.seed(42)
    trend = np.linspace(100, 150, n)
    noise = np.random.normal(0, 2, n)
    close = trend + noise
    high = close + np.abs(np.random.normal(1, 0.5, n))
    low = close - np.abs(np.random.normal(1, 0.5, n))
    open_price = close + np.random.normal(0, 1, n)
    volume = np.random.uniform(1000000, 10000000, n)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Test ZigZag
    zigzag = ZigZagIndicator(depth=5, deviation=2, backstep=2)
    df_labeled = zigzag.label_kbars(df)
    
    print(f'  Shape: {df_labeled.shape}')
    print(f'  Columns: {df_labeled.columns.tolist()}')
    print(f'  Label distribution: {df_labeled["zigzag_label"].value_counts().to_dict()}')
    print(f'  Success!')
    return df_labeled

def test_feature_engineering():
    """
    Test feature engineering pipeline.
    """
    print('\nTest 2: Feature engineering')
    
    # Use data from previous test
    n = 200
    np.random.seed(42)
    trend = np.linspace(100, 150, n)
    noise = np.random.normal(0, 2, n)
    close = trend + noise
    high = close + np.abs(np.random.normal(1, 0.5, n))
    low = close - np.abs(np.random.normal(1, 0.5, n))
    open_price = close + np.random.normal(0, 1, n)
    volume = np.random.uniform(1000000, 10000000, n)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Test feature engineering
    fe = FeatureEngineer(lookback_periods=[5, 10, 20])
    df_features = fe.calculate_all_features(df)
    
    print(f'  Shape: {df_features.shape}')
    feature_cols = fe.get_feature_columns(df_features)
    print(f'  Number of features: {len(feature_cols)}')
    print(f'  Sample features: {feature_cols[:10]}')
    print(f'  Missing values: {df_features[feature_cols].isnull().sum().sum()}')
    print(f'  Success!')
    return df_features

if __name__ == '__main__':
    print('Running tests...\n')
    test_zigzag_basic()
    test_feature_engineering()
    print('\nAll tests passed!')
