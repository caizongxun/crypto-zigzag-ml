#!/usr/bin/env python
"""
Parallel Training Worker - 單個進程負責一個 symbol+timeframe 組合
使用方式: python train_worker.py 0 BTCUSDT 15m
"""
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# 設定 GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if len(sys.argv) > 1:
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    except:
        pass

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import precision_score, recall_score, f1_score

# 你的模組 (修改為實際路徑)
from data.fetch_data import CryptoDataFetcher
from src.zigzag_indicator import ZigZagIndicator
from src.features import FeatureEngineer
from src.utils import time_series_split

# 設定 GPU 記憶體增長
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass

CONFIG = {
    'features': 20,
    'timesteps': 10,
    'batch_size': 64,
    'clf_epochs': 100,
    'det_epochs': 80,
    'early_stop_patience': 8,
}

def create_sequences(X, y, timesteps=10):
    """建立時間序列樣本"""
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:(i + timesteps)])
        y_seq.append(y[i + timesteps])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq)

def build_clf_model(input_shape):
    """建立 5 類分類模型"""
    model = keras.Sequential([
        layers.LSTM(128, input_shape=input_shape, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=False),
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

def build_det_model(input_shape):
    """建立二元檢測模型"""
    model = keras.Sequential([
        layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_symbol_timeframe(symbol, timeframe, project_root):
    """訓練單個 symbol+timeframe 組合"""
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    print(f'[GPU{gpu_id}] Starting {symbol} {timeframe} at {datetime.now().strftime("%H:%M:%S")}')
    
    try:
        # 初始化
        fetcher = CryptoDataFetcher()
        zigzag = ZigZagIndicator(depth=12, deviation=5, backstep=2)
        fe = FeatureEngineer(lookback_periods=[5, 10, 20, 50, 200])
        
        # 取得數據
        print(f'[GPU{gpu_id}]   Fetching data...')
        data = fetcher.fetch_symbol_timeframe(symbol, timeframe)
        if len(data) < 500:
            print(f'[GPU{gpu_id}] ✗ {symbol} {timeframe}: Insufficient data ({len(data)} bars)')
            return None
        
        # ZigZag 標籤
        print(f'[GPU{gpu_id}]   Applying ZigZag...')
        data = zigzag.label_kbars(data)
        
        # 特徵工程
        print(f'[GPU{gpu_id}]   Engineering features...')
        data = fe.calculate_all_features(data)
        feature_cols = fe.get_feature_columns(data)
        data[feature_cols] = data[feature_cols].fillna(method='ffill').fillna(0)
        
        # 時間序列分割
        train_df, val_df, test_df = time_series_split(data, 0.7, 0.15)
        
        selected_features = feature_cols[:CONFIG['features']]
        
        # 準備數據
        X_train = train_df[selected_features].values.astype(np.float32)
        y_train = train_df['zigzag_label'].values
        X_val = val_df[selected_features].values.astype(np.float32)
        y_val = val_df['zigzag_label'].values
        X_test = test_df[selected_features].values.astype(np.float32)
        y_test = test_df['zigzag_label'].values
        
        # 標準化
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std
        
        # 創建序列
        print(f'[GPU{gpu_id}]   Creating sequences...')
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, CONFIG['timesteps'])
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, CONFIG['timesteps'])
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, CONFIG['timesteps'])
        
        # 類別權重
        unique, counts = np.unique(y_train_seq, return_counts=True)
        total = len(y_train_seq)
        class_weights = {}
        for u, c in zip(unique, counts):
            class_weights[u] = 1.0 if u == 0 else total / (5 * c) * 3
        
        # 二元標籤
        y_train_binary = (y_train_seq != 0).astype(np.float32)
        y_val_binary = (y_val_seq != 0).astype(np.float32)
        y_test_binary = (y_test_seq != 0).astype(np.float32)
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['early_stop_patience'],
            restore_best_weights=True,
            verbose=0
        )
        
        # 訓練分類模型
        print(f'[GPU{gpu_id}]   Training classification...')
        clf_model = build_clf_model((X_train_seq.shape[1], X_train_seq.shape[2]))
        clf_model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=CONFIG['clf_epochs'],
            batch_size=CONFIG['batch_size'],
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=0
        )
        clf_loss, clf_acc = clf_model.evaluate(X_test_seq, y_test_seq, verbose=0)
        
        # 訓練檢測模型
        print(f'[GPU{gpu_id}]   Training detection...')
        det_model = build_det_model((X_train_seq.shape[1], X_train_seq.shape[2]))
        det_model.fit(
            X_train_seq, y_train_binary,
            validation_data=(X_val_seq, y_val_binary),
            epochs=CONFIG['det_epochs'],
            batch_size=CONFIG['batch_size'],
            callbacks=[early_stop],
            verbose=0
        )
        det_loss, det_acc = det_model.evaluate(X_test_seq, y_test_binary, verbose=0)
        
        # 保存模型
        print(f'[GPU{gpu_id}]   Saving models...')
        model_dir = f'{project_root}/models/{symbol.lower()}_{timeframe}'
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        clf_model.save(f'{model_dir}/classification.h5')
        det_model.save(f'{model_dir}/detection.h5')
        
        params = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'metrics': {'clf_acc': float(clf_acc), 'det_acc': float(det_acc)},
            'normalization': {'mean': mean.tolist(), 'std': std.tolist()},
            'class_weights': {int(k): v for k, v in class_weights.items()}
        }
        with open(f'{model_dir}/params.json', 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f'[GPU{gpu_id}] ✓ {symbol} {timeframe}: clf={clf_acc:.4f}, det={det_acc:.4f}')
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'clf_acc': float(clf_acc),
            'det_acc': float(det_acc),
            'status': 'ok'
        }
    
    except Exception as e:
        error_msg = str(e)[:200]
        print(f'[GPU{gpu_id}] ✗ {symbol} {timeframe}: {error_msg}')
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'error',
            'error': error_msg
        }

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python train_worker.py <gpu_id> <symbol> <timeframe>')
        print('Example: python train_worker.py 0 BTCUSDT 15m')
        sys.exit(1)
    
    gpu_id = sys.argv[1]
    symbol = sys.argv[2]
    timeframe = sys.argv[3]
    
    # 修改為你的專案路徑
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    result = train_symbol_timeframe(symbol, timeframe, project_root)
    
    if result:
        print(json.dumps(result))
        sys.exit(0 if result['status'] == 'ok' else 1)
    else:
        sys.exit(1)
