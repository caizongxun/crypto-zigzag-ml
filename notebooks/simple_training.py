"""
Simple training script - standalone version without complex dependencies
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path('.').resolve().parent
sys.path.insert(0, str(project_root))

from data.fetch_data import CryptoDataFetcher
from src.zigzag_indicator import ZigZagIndicator
from src.features import FeatureEngineer

# ============================================================================
# SIMPLE NORMALIZATION (no NumPy issues)
# ============================================================================
def simple_normalize(X):
    """Simple Z-score normalization that avoids NumPy issues"""
    X = np.asarray(X, dtype=np.float64)
    
    # Calculate mean and std safely
    mean = np.mean(X, axis=0)
    
    # Subtract mean
    X_centered = X - mean
    
    # Calculate variance manually
    var = np.mean(X_centered ** 2, axis=0)
    std = np.sqrt(var + 1e-8)
    
    # Normalize
    X_norm = X_centered / std
    
    return X_norm, {'mean': mean, 'std': std}

def apply_normalization(X, params):
    """Apply saved normalization parameters"""
    X = np.asarray(X, dtype=np.float64)
    mean = params['mean']
    std = params['std']
    return (X - mean) / std

# ============================================================================
# DATA PREPARATION
# ============================================================================
print("=" * 80)
print("STEP 1: Fetching Data")
print("=" * 80)

fetcher = CryptoDataFetcher()
btc_15m = fetcher.fetch_symbol_timeframe('BTCUSDT', '15m')
print(f"Data fetched. Shape: {btc_15m.shape}")

# ============================================================================
# APPLY ZIGZAG
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Applying ZigZag and Labeling")
print("=" * 80)

zigzag = ZigZagIndicator(depth=12, deviation=5, backstep=2)
btc_15m = zigzag.label_kbars(btc_15m)

label_counts = btc_15m['zigzag_label'].value_counts().sort_index()
print("Label Distribution:")
for label_id, count in label_counts.items():
    label_name = zigzag.get_label_name(label_id)
    print(f"  {label_name}: {count}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Feature Engineering")
print("=" * 80)

fe = FeatureEngineer(lookback_periods=[5, 10, 20, 50, 200])
btc_15m = fe.calculate_all_features(btc_15m)

feature_cols = fe.get_feature_columns(btc_15m)
btc_15m[feature_cols] = btc_15m[feature_cols].fillna(method='ffill').fillna(0)

print(f"Total features: {len(feature_cols)}")
print(f"Sample features: {feature_cols[:10]}")

# Remove non-numeric columns  
numeric_feature_cols = [col for col in feature_cols 
                       if col not in ['symbol', 'timestamp', 'date']
                       and btc_15m[col].dtype in [np.float64, np.float32, np.int64, np.int32]]

print(f"Numeric features: {len(numeric_feature_cols)}")

# ============================================================================
# TIME SERIES SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Time Series Split")
print("=" * 80)

n = len(btc_15m)
train_size = int(n * 0.7)
val_size = int(n * 0.15)

train_df = btc_15m.iloc[:train_size]
val_df = btc_15m.iloc[train_size:train_size + val_size]
test_df = btc_15m.iloc[train_size + val_size:]

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ============================================================================
# PREPARE DATA FOR TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Preparing Sequences for LSTM")
print("=" * 80)

X_train = train_df[numeric_feature_cols].values
y_train = train_df['zigzag_label'].values
X_val = val_df[numeric_feature_cols].values
y_val = val_df['zigzag_label'].values
X_test = test_df[numeric_feature_cols].values
y_test = test_df['zigzag_label'].values

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_train unique labels: {np.unique(y_train)}")

# Normalize using simple method
print("\nNormalizing data...")
X_train_norm, norm_params = simple_normalize(X_train)
X_val_norm = apply_normalization(X_val, norm_params)
X_test_norm = apply_normalization(X_test, norm_params)

print(f"Normalization complete")
print(f"X_train_norm mean: {X_train_norm.mean():.4f}, std: {X_train_norm.std():.4f}")

# ============================================================================
# CREATE SEQUENCES FOR LSTM
# ============================================================================
def create_sequences(X, y, timesteps=60):
    """Create sequences for LSTM"""
    X_seq, y_seq = [], []
    
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:(i + timesteps)])
        y_seq.append(y[i + timesteps])
    
    return np.array(X_seq), np.array(y_seq)

print("\nCreating sequences (timesteps=60)...")
X_train_seq, y_train_seq = create_sequences(X_train_norm, y_train, timesteps=60)
X_val_seq, y_val_seq = create_sequences(X_val_norm, y_val, timesteps=60)
X_test_seq, y_test_seq = create_sequences(X_test_norm, y_test, timesteps=60)

print(f"Train sequences: {X_train_seq.shape}, labels: {y_train_seq.shape}")
print(f"Val sequences: {X_val_seq.shape}, labels: {y_val_seq.shape}")
print(f"Test sequences: {X_test_seq.shape}, labels: {y_test_seq.shape}")

# ============================================================================
# DATA SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING DATA SUMMARY")
print("=" * 80)
print(f"Feature count: {X_train_seq.shape[2]}")
print(f"Sequence length: {X_train_seq.shape[1]}")
print(f"Training samples: {X_train_seq.shape[0]}")
print(f"Validation samples: {X_val_seq.shape[0]}")
print(f"Test samples: {X_test_seq.shape[0]}")

# Label distribution
print("\nLabel distribution in training data:")
unique, counts = np.unique(y_train_seq, return_counts=True)
for label, count in zip(unique, counts):
    pct = (count / len(y_train_seq)) * 100
    print(f"  Label {label}: {count:6d} ({pct:5.2f}%)")

print("\n" + "=" * 80)
print("DATA PREPARATION COMPLETE!")
print("=" * 80)
print("\nNext steps:")
print("1. Run model training (train LSTM + XGBoost)")
print("2. Evaluate on test set")
print("3. Deploy to production")
print("\nVariables ready for model training:")
print(f"  - X_train_seq, y_train_seq")
print(f"  - X_val_seq, y_val_seq")
print(f"  - X_test_seq, y_test_seq")
print(f"  - norm_params (for denormalization)")
