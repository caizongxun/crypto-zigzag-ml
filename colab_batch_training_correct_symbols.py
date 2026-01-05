#!/usr/bin/env python
"""
Smart Batch Training - Only trains missing classifier models
With CORRECT symbol list from Hugging Face

正確的 18 個幣種 (來自你的圖):
BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT,
DOGEUSDT, MATICUSDT, LINKUSDT, LITUSDT, UNIUSDT,
AVAXUSDT, SOLUUSDT, FTMUSDT, CRVUSDT, MKRUSDT,
SNXUSDT, COMPUSDT, LRCUSDT, GRTUSDT

Logic:
1. 讀取 Hugging Face 已有的模型
2. 檢查本地 Google Drive 已有的模型
3. 只訓練缺少的模型
4. 自動上傳新訓練的模型到 HF

Usage in Colab:
1. exec(open('colab_batch_training_correct_symbols.py').read())
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

print('Preparing smart training environment...')
print('='*70)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

print(f'TensorFlow: {tf.__version__}')
print(f'GPU: {tf.config.list_physical_devices("GPU")}')
print()

# Check HF availability
try:
    from huggingface_hub import hf_hub_url, HfApi, CommitOperationAdd
    print('Hugging Face hub available')
    HF_AVAILABLE = True
except:
    print('Warning: huggingface-hub not available (will skip HF checks)')
    HF_AVAILABLE = False

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

print()
print('='*70)
print('SMART BATCH TRAINING - CLASSIFIER MODELS')
print('='*70)
print()

# Configuration - CORRECTED SYMBOL LIST
# 18 symbols with 2 timeframes = 36 models total
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    'DOGEUSDT', 'MATICUSDT', 'LINKUSDT', 'LITUSDT', 'UNIUSDT',
    'AVAXUSDT', 'SOLUUSDT', 'FTMUSDT', 'CRVUSDT', 'MKRUSDT',
    'SNXUSDT', 'COMPUSDT', 'LRCUSDT', 'GRTUSDT'
]

TIMEFRAMES = ['15m', '1h']
HF_REPO_ID = 'zongowo111/v2-crypto-ohlcv-data'

print(f'Symbols: {len(SYMBOLS)} (from your list)')
print(f'Timeframes: {len(TIMEFRAMES)}')
print(f'Total models target: {len(SYMBOLS) * len(TIMEFRAMES)}')
print()
print('Symbols:')
for i, sym in enumerate(SYMBOLS, 1):
    print(f'  {i:2d}. {sym}')
print()

CONFIG = {
    'features': 20,
    'timesteps': 10,
    'batch_size': 64,
    'epochs': 100,
    'early_stop_patience': 8,
    'lstm_layers': [128, 64],
}

def get_hf_models():
    """
    Get list of already trained models from Hugging Face
    """
    if not HF_AVAILABLE:
        return set()
    
    hf_models = set()
    try:
        from huggingface_hub import list_repo_tree
        print('Checking Hugging Face for existing models...')
        
        # List all files in v1_model directory
        files = list_repo_tree(HF_REPO_ID, recursive=True, repo_type='dataset')
        
        for file_info in files:
            path = file_info.path
            # Parse: v1_model/SYMBOL/TIMEFRAME/classification.h5
            if 'v1_model' in path and 'classification.h5' in path:
                parts = path.split('/')
                if len(parts) >= 4:
                    symbol = parts[1]  # SYMBOL
                    timeframe = parts[2]  # TIMEFRAME
                    hf_models.add((symbol, timeframe))
        
        print(f'Found {len(hf_models)} models on HF')
        if hf_models:
            for symbol, timeframe in sorted(hf_models)[:5]:
                print(f'  ✓ {symbol} {timeframe}')
            if len(hf_models) > 5:
                print(f'  ... and {len(hf_models)-5} more')
    
    except Exception as e:
        print(f'Warning: Could not list HF models: {e}')
    
    return hf_models

def get_local_models(models_dir):
    """
    Get list of models already in Google Drive
    """
    local_models = set()
    models_path = Path(models_dir)
    
    if not models_path.exists():
        return local_models
    
    for model_dir in models_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Parse: btcusdt_15m → symbol=BTCUSDT, timeframe=15m
        parts = model_dir.name.rsplit('_', 1)
        if len(parts) == 2:
            symbol, timeframe = parts[0].upper(), parts[1]
            h5_file = model_dir / 'classification.h5'
            if h5_file.exists():
                local_models.add((symbol, timeframe))
    
    return local_models

def get_models_to_train(symbols, timeframes):
    """
    Determine which models need to be trained
    """
    models_dir = f'{project_root}/models'
    
    # Get existing models
    hf_models = get_hf_models()
    local_models = get_local_models(models_dir)
    
    print()
    print(f'Local models: {len(local_models)}')
    if local_models:
        for symbol, timeframe in sorted(local_models)[:5]:
            print(f'  ✓ {symbol} {timeframe}')
        if len(local_models) > 5:
            print(f'  ... and {len(local_models)-5} more')
    
    print()
    
    # All existing models (local + HF)
    all_existing = hf_models | local_models
    
    # Models to train
    to_train = []
    for symbol in symbols:
        for timeframe in timeframes:
            if (symbol, timeframe) not in all_existing:
                to_train.append((symbol, timeframe))
    
    print(f'Models to train: {len(to_train)}')
    print(f'Total progress: {len(all_existing)}/{len(symbols)*len(timeframes)}')
    print()
    
    return to_train, all_existing

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

def upload_model_to_hf(symbol, timeframe, model_dir, hf_token):
    """
    Upload a trained model to Hugging Face
    """
    if not HF_AVAILABLE or not hf_token:
        return False
    
    try:
        from huggingface_hub import HfApi, CommitOperationAdd
        api = HfApi(token=hf_token)
        
        h5_file = model_dir / 'classification.h5'
        params_file = model_dir / 'params.json'
        
        if not h5_file.exists() or not params_file.exists():
            return False
        
        h5_hf_path = f'v1_model/{symbol}/{timeframe}/classification.h5'
        params_hf_path = f'v1_model/{symbol}/{timeframe}/params.json'
        
        operations = [
            CommitOperationAdd(
                path_in_repo=h5_hf_path,
                path_or_fileobj=h5_file
            ),
            CommitOperationAdd(
                path_in_repo=params_hf_path,
                path_or_fileobj=params_file
            )
        ]
        
        api.create_commit(
            repo_id=HF_REPO_ID,
            repo_type='dataset',
            operations=operations,
            commit_message=f'Add {symbol} {timeframe} classification model'
        )
        
        return True
    except Exception as e:
        print(f'  Upload error: {e}')
        return False

def train_clf_model(symbol, timeframe, fetcher, zigzag, fe, hf_token=None):
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
        
        # Upload to HF if token available
        uploaded = False
        if hf_token:
            uploaded = upload_model_to_hf(symbol, timeframe, Path(model_dir), hf_token)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'ok',
            'acc': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'uploaded': uploaded
        }
    
    except Exception as e:
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'error',
            'error': str(e)[:100]
        }

# Main execution
print('Checking training status...')
print()

models_to_train, already_trained = get_models_to_train(SYMBOLS, TIMEFRAMES)

if not models_to_train:
    print('='*70)
    print(f'All {len(SYMBOLS) * len(TIMEFRAMES)} models already trained!')
    print('='*70)
    sys.exit(0)

print('='*70)
print(f'TRAINING {len(models_to_train)} REMAINING MODELS')
print('='*70)
print()

# Ask about HF token
hf_token = None
if HF_AVAILABLE:
    try:
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            response = input('Auto-upload to Hugging Face? (y/n): ')
            if response.lower() == 'y':
                hf_token = input('Enter HF token (optional, press Enter to skip): ')
                if not hf_token.strip():
                    hf_token = None
    except:
        pass

print()

# Training loop
fetcher = CryptoDataFetcher()
zigzag = ZigZagIndicator(depth=12, deviation=5, backstep=2)
fe = FeatureEngineer(lookback_periods=[5, 10, 20, 50, 200])

results = []
start_time = datetime.now()

for idx, (symbol, timeframe) in enumerate(models_to_train, 1):
    total_progress = len(already_trained) + idx
    total_target = len(SYMBOLS) * len(TIMEFRAMES)
    print(f'[{total_progress:2d}/{total_target}] {symbol:10s} {timeframe}... ', end='', flush=True)
    
    result = train_clf_model(symbol, timeframe, fetcher, zigzag, fe, hf_token)
    results.append(result)
    
    if result['status'] == 'ok':
        uploaded_mark = ' (uploaded to HF)' if result.get('uploaded') else ''
        print(f'✓ f1={result["f1"]:.4f}{uploaded_mark}')
    elif result['status'] == 'skip':
        print('⊘ skip')
    else:
        print('✗ error')

print()
print('='*70)

# Summary
total_time = (datetime.now() - start_time).total_seconds() / 3600
successful = [r for r in results if r['status'] == 'ok']
failed = [r for r in results if r['status'] == 'error']
skipped = [r for r in results if r['status'] == 'skip']
uploaded = [r for r in results if r.get('uploaded')]

print('BATCH TRAINING SUMMARY')
print('='*70)
print(f'Total time: {total_time:.1f} hours')
print(f'Newly trained: {len(successful)}/{len(models_to_train)}')
print(f'Failed: {len(failed)}')
print(f'Skipped: {len(skipped)}')
print(f'Uploaded to HF: {len(uploaded)}')
print(f'Overall progress: {len(already_trained) + len(successful)}/{len(SYMBOLS)*len(TIMEFRAMES)}')

if successful:
    avg_f1 = np.mean([r['f1'] for r in successful])
    print(f'\nNew models average F1: {avg_f1:.4f}')
    
    sorted_by_f1 = sorted(successful, key=lambda x: x['f1'], reverse=True)[:5]
    print(f'\nTop 5 new models:')
    for r in sorted_by_f1:
        print(f'  {r["symbol"]:10s} {r["timeframe"]:3s}: F1={r["f1"]:.4f}')

if failed:
    print(f'\nFailed models:')
    for r in failed:
        print(f'  {r["symbol"]:10s} {r["timeframe"]:3s}')

print('='*70)
print('Training complete!')
print('='*70)
