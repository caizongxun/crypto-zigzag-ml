#!/usr/bin/env python
"""
Colab Batch Training - 44 Classification Models

Usage in Colab:
1. Upload this file to Colab
2. Run: exec(open('colab_batch_training.py').read())
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

print('Preparing Colab environment...')
print('='*70)

# Setup
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

print(f'TensorFlow: {tf.__version__}')
print(f'GPU: {tf.config.list_physical_devices("GPU")}')
print()

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    project_root = '/content/drive/MyDrive/crypto-zigzag-ml'
    print(f'Google Drive mounted')
except:
    project_root = '.'
    print('Not in Colab, using local directory')

print()

# Import custom modules
sys.path.insert(0, project_root)
try:
    from data.fetch_data import CryptoDataFetcher
    from src.zigzag_indicator import ZigZagIndicator
    from src.features import FeatureEngineer
    from src.utils import time_series_split
    print('Custom modules imported successfully')
except Exception as e:
    print(f'Warning: Could not import custom modules: {e}')
    print('Make sure data/, src/ directories are uploaded to Google Drive')

print()
print('='*70)
print('BATCH TRAINING - 44 CLASSIFICATION MODELS')
print('='*70)
print()

# Configuration
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    'DOGEUSDT', 'MATICUSDT', 'LINKUSDT', 'LITUSDT', 'UNIUSDT',
    'AVAXUSDT', 'SOLUUSDT', 'FTMUSDT', 'AAVEUSDT', 'CRVUSDT',
    'MKRUSDT', 'SNXUSDT', 'COMPUSDT', 'LRCUSDT', 'GRTUSDT',
    'ALGOUSDT', 'ATOMUSDT'
]

TIMEFRAMES = ['15m', '1h']

CONFIG = {
    'features': 20,
    'timesteps': 10,
    'batch_size': 64,
    'epochs': 100,
    'early_stop_patience': 8,
    'lstm_layers': [128, 64],
}

print(f'Symbols: {len(SYMBOLS)}')
print(f'Timeframes: {len(TIMEFRAMES)}')
print(f'Total models: {len(SYMBOLS) * len(TIMEFRAMES)}')
print(f'Estimated time: {len(SYMBOLS) * len(TIMEFRAMES) * 0.3:.1f} hours')
print()

# Helper functions
def create_sequences(X, y, timesteps=10):
    """Create time series sequences"""
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:(i + timesteps)])
        y_seq.append(y[i + timesteps])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq)

def build_clf_model(input_shape, lstm_layers=[128, 64]):
    """Build LSTM classification model"""
    model = keras.Sequential([
        layers.LSTM(lstm_layers[0], input_shape=input_shape, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(lstm_layers[1], return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(5, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_clf_model(symbol, timeframe, fetcher, zigzag, fe):
    """Train a single classification model"""
    try:
        # Data
        data = fetcher.fetch_symbol_timeframe(symbol, timeframe)
        if len(data) < 500:
            return {'symbol': symbol, 'timeframe': timeframe, 'status': 'skip'}
        
        # ZigZag
        data = zigzag.label_kbars(data)
        
        # Features
        data = fe.calculate_all_features(data)
        feature_cols = fe.get_feature_columns(data)
        data[feature_cols] = data[feature_cols].fillna(method='ffill').fillna(0)
        
        # Split
        train_df, val_df, test_df = time_series_split(data, 0.7, 0.15)
        selected_features = feature_cols[:CONFIG['features']]
        
        X_train = train_df[selected_features].values.astype(np.float32)
        y_train = train_df['zigzag_label'].values
        X_val = val_df[selected_features].values.astype(np.float32)
        y_val = val_df['zigzag_label'].values
        X_test = test_df[selected_features].values.astype(np.float32)
        y_test = test_df['zigzag_label'].values
        
        # Normalize
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std
        
        # Sequences
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, CONFIG['timesteps'])
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, CONFIG['timesteps'])
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, CONFIG['timesteps'])
        
        # Class weights
        unique, counts = np.unique(y_train_seq, return_counts=True)
        total = len(y_train_seq)
        class_weights = {}
        for u, c in zip(unique, counts):
            class_weights[u] = 1.0 if u == 0 else total / (5 * c) * 3
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['early_stop_patience'],
            restore_best_weights=True,
            verbose=0
        )
        
        # Train
        model = build_clf_model((X_train_seq.shape[1], X_train_seq.shape[2]))
        model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        loss, acc = model.evaluate(X_test_seq, y_test_seq, verbose=0)
        y_pred = np.argmax(model.predict(X_test_seq, verbose=0), axis=1)
        precision = precision_score(y_test_seq, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_seq, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_seq, y_pred, average='weighted', zero_division=0)
        
        # Save
        model_dir = f'{project_root}/models/{symbol.lower()}_{timeframe}/'
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model.save(f'{model_dir}classification.h5')
        
        params = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'metrics': {'acc': float(acc), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1)},
            'normalization': {'mean': mean.tolist(), 'std': std.tolist()},
            'class_weights': {int(k): v for k, v in class_weights.items()}
        }
        with open(f'{model_dir}params.json', 'w') as f:
            json.dump(params, f, indent=2)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'ok',
            'acc': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    except Exception as e:
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'error',
            'error': str(e)[:100]
        }

# Training loop
fetcher = CryptoDataFetcher()
zigzag = ZigZagIndicator(depth=12, deviation=5, backstep=2)
fe = FeatureEngineer(lookback_periods=[5, 10, 20, 50, 200])

results = []
start_time = datetime.now()

for i, symbol in enumerate(SYMBOLS, 1):
    for j, timeframe in enumerate(TIMEFRAMES, 1):
        model_num = (i-1) * len(TIMEFRAMES) + j
        print(f'[{model_num:2d}/44] {symbol:10s} {timeframe}... ', end='', flush=True)
        
        result = train_clf_model(symbol, timeframe, fetcher, zigzag, fe)
        results.append(result)
        
        if result['status'] == 'ok':
            print(f'✓ acc={result["acc"]:.4f} f1={result["f1"]:.4f}')
        elif result['status'] == 'skip':
            print('⊘ skip')
        else:
            print(f'✗ error')

print()
print('='*70)

# Summary
total_time = (datetime.now() - start_time).total_seconds() / 3600
successful = [r for r in results if r['status'] == 'ok']
failed = [r for r in results if r['status'] == 'error']
skipped = [r for r in results if r['status'] == 'skip']

print('BATCH TRAINING SUMMARY')
print('='*70)
print(f'Total time: {total_time:.1f} hours')
print(f'Completed: {len(successful)}/{len(results)}')
print(f'Failed: {len(failed)}')
print(f'Skipped: {len(skipped)}')

if successful:
    avg_acc = np.mean([r['acc'] for r in successful])
    avg_f1 = np.mean([r['f1'] for r in successful])
    
    print(f'\nAverage Metrics:')
    print(f'  Accuracy: {avg_acc:.4f}')
    print(f'  F1-Score: {avg_f1:.4f}')
    
    sorted_by_f1 = sorted(successful, key=lambda x: x['f1'], reverse=True)[:5]
    print(f'\nTop 5 Models:')
    for r in sorted_by_f1:
        print(f'  {r["symbol"]:10s} {r["timeframe"]:3s}: F1={r["f1"]:.4f}')

if failed:
    print(f'\nFailed Models:')
    for r in failed:
        print(f'  {r["symbol"]:10s} {r["timeframe"]:3s}')

print('='*70)
print(f'Models saved to: {project_root}/models/')
print('='*70)
print('\nTraining complete!')
