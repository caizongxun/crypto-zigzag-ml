#!/usr/bin/env python3
"""
Helper script to fix Stage 2 training with proper 3D sequence conversion.
Run this before the main training to verify the pipeline works correctly.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

def create_3d_sequences(X, y, seq_length=10):
    """
    Convert 2D features to 3D sequences for LSTM/CNN models.
    
    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        seq_length: Window length for sequences (default: 10)
    
    Returns:
        X_seq: 3D sequences (n_sequences, seq_length, n_features)
        y_seq: Corresponding labels (n_sequences,)
    """
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int32)


def validate_stage1_model_input(model, X_3d):
    """
    Validate that 3D data matches Stage 1 model requirements.
    
    Args:
        model: Keras model
        X_3d: 3D feature array (n_sequences, seq_len, n_features)
    
    Returns:
        bool: True if compatible
    """
    input_shape = model.input_shape
    print(f"Model expected input shape: {input_shape}")
    print(f"Data shape: {X_3d.shape}")
    
    if len(X_3d.shape) != 3:
        print(f"ERROR: Expected 3D input (batch, seq, features), got {len(X_3d.shape)}D")
        return False
    
    batch_size, seq_len, n_features = X_3d.shape
    expected_seq_len = input_shape[1]
    expected_n_features = input_shape[2]
    
    if seq_len != expected_seq_len:
        print(f"ERROR: Sequence length mismatch. Got {seq_len}, expected {expected_seq_len}")
        return False
    
    if n_features != expected_n_features:
        print(f"ERROR: Number of features mismatch. Got {n_features}, expected {expected_n_features}")
        return False
    
    print("✓ Input shape validated successfully")
    return True


def test_stage1_inference(model, X_3d, batch_size=32):
    """
    Test Stage 1 model inference on 3D data.
    
    Args:
        model: Keras model
        X_3d: 3D feature array
        batch_size: Inference batch size
    
    Returns:
        predictions: Model predictions
    """
    print(f"\nTesting Stage 1 inference...")
    print(f"  Input shape: {X_3d.shape}")
    print(f"  Batch size: {batch_size}")
    
    try:
        predictions = model.predict(X_3d, batch_size=batch_size, verbose=0)
        print(f"  Output shape: {predictions.shape}")
        print(f"  Output range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print("✓ Inference successful")
        return predictions
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        raise


def demonstrate_pipeline(df, feature_cols, seq_length=10):
    """
    Demonstrate the complete pipeline for Stage 2 training.
    
    Args:
        df: DataFrame with zigzag_label column
        feature_cols: List of feature column names
        seq_length: Sequence length for 3D conversion
    """
    print("\n" + "="*80)
    print("STAGE 2 TRAINING PIPELINE DEMONSTRATION")
    print("="*80)
    
    # Extract features and labels
    X = df[feature_cols].values
    y = df['zigzag_label'].values
    
    print(f"\nOriginal Data:")
    print(f"  Shape: {X.shape} (samples={X.shape[0]}, features={X.shape[1]})")
    print(f"  Label distribution: {np.bincount(y)}")
    
    # Create 3D sequences
    print(f"\nConverting to 3D sequences (seq_length={seq_length})...")
    X_3d, y_3d = create_3d_sequences(X, y, seq_length=seq_length)
    
    print(f"  Output shape: {X_3d.shape}")
    print(f"  Expected for Stage 1: (batch, {seq_length}, {X.shape[1]})")
    print(f"✓ Conversion successful")
    
    # Statistics
    print(f"\nSequence Statistics:")
    print(f"  Total sequences: {len(X_3d):,}")
    print(f"  Reduction from original: {len(X) - len(X_3d)} samples")
    print(f"  Label distribution in sequences:")
    for label in np.unique(y_3d):
        count = (y_3d == label).sum()
        pct = 100 * count / len(y_3d)
        print(f"    Label {label}: {count:,} ({pct:.1f}%)")
    
    return X_3d, y_3d


if __name__ == "__main__":
    print("Stage 2 Training Fix Script")
    print("This script validates the 3D sequence conversion pipeline.\n")
    
    # Check imports
    try:
        from tensorflow import keras
        print("✓ TensorFlow imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TensorFlow: {e}")
        sys.exit(1)
    
    print("\nKey Functions Available:")
    print("  1. create_3d_sequences(X, y, seq_length=10)")
    print("     → Converts 2D features to 3D time series")
    print("  2. validate_stage1_model_input(model, X_3d)")
    print("     → Validates compatibility with Stage 1 model")
    print("  3. test_stage1_inference(model, X_3d)")
    print("     → Tests model inference")
    print("  4. demonstrate_pipeline(df, feature_cols, seq_length=10)")
    print("     → Full pipeline demonstration")
    
    print("\n" + "="*80)
    print("Usage in your notebook:")
    print("="*80)
    print("""
from scripts.fix_stage2_training import create_3d_sequences, validate_stage1_model_input

# Convert features to 3D
X_train_3d, y_train_3d = create_3d_sequences(X_train_2d, y_train_2d, seq_length=10)

# Validate with model
validate_stage1_model_input(stage1_model, X_train_3d)

# Use with model
stage1_probs = stage1_model.predict(X_train_3d, verbose=0)  # Now works!
    """)
