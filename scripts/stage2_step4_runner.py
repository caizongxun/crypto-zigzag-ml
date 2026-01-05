#!/usr/bin/env python3
"""
Stage 2 Step 4 Runner - 獨立執行腳本

用途：在 Colab 中直接執行完整的 Step 4 流程（數據分割 + 3D 轉換 + Stage 1 過濾）

使用方法：
  # 在 Colab Cell 中執行
  !curl -s https://raw.githubusercontent.com/caizongxun/crypto-zigzag-ml/main/scripts/stage2_step4_runner.py | python3

或者：
  !wget -q https://raw.githubusercontent.com/caizongxun/crypto-zigzag-ml/main/scripts/stage2_step4_runner.py
  !python3 stage2_step4_runner.py

前提條件：
  1. 已執行 Step 1: 下載 Stage 1 模型
  2. 已執行 Step 2: 下載訓練數據
  3. 已執行 Step 3: 特徵工程
  4. 以下變數已定義：
     - df: DataFrame with zigzag_label column
     - feature_cols: list of feature column names
     - stage1_model: Loaded Keras model
     - STAGE1_SEQUENCE_LENGTH: int (usually 10)
     - STAGE2_DATA_DIR: Path for saving data (optional)
     - time_series_split: function for splitting (optional)
"""

import sys
import os
import numpy as np
import pickle
from pathlib import Path


def create_3d_sequences(X, y, seq_length=10):
    """
    Convert 2D features to 3D sequences for LSTM/CNN models.
    
    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        seq_length: Window length for sequences
    
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


def get_ipython_namespace():
    """
    Get the IPython user namespace if available, otherwise return None.
    """
    try:
        import IPython
        ipython = IPython.get_ipython()
        if ipython is not None:
            return ipython.user_ns
    except (ImportError, AttributeError):
        pass
    return None


def validate_inputs(user_ns):
    """
    Validate that all required inputs are available.
    """
    print("驗證輸入...")
    
    # Get from namespace
    df = user_ns.get('df')
    feature_cols = user_ns.get('feature_cols')
    stage1_model = user_ns.get('stage1_model')
    STAGE1_SEQUENCE_LENGTH = user_ns.get('STAGE1_SEQUENCE_LENGTH', 10)
    time_series_split = user_ns.get('time_series_split')
    
    # Check df
    if df is None:
        raise ValueError("df 未定義。請確保已執行 Step 1-3")
    if not isinstance(df, object) or not hasattr(df, 'shape'):
        raise ValueError("df 必須是 DataFrame")
    if 'zigzag_label' not in df.columns:
        raise ValueError("df 必須包含 'zigzag_label' 列")
    print(f"  ✓ df shape: {df.shape}")
    
    # Check feature_cols
    if feature_cols is None:
        raise ValueError("feature_cols 未定義")
    if not isinstance(feature_cols, list):
        raise ValueError("feature_cols 必須是 list")
    if len(feature_cols) == 0:
        raise ValueError("feature_cols 為空")
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"特徵列 '{col}' 不在 df 中")
    print(f"  ✓ feature_cols: {len(feature_cols)} features")
    
    # Check stage1_model
    if stage1_model is None:
        raise ValueError("stage1_model 未載入。請確保已執行 Step 1")
    input_shape = stage1_model.input_shape
    print(f"  ✓ stage1_model input shape: {input_shape}")
    
    # Check STAGE1_SEQUENCE_LENGTH
    if not isinstance(STAGE1_SEQUENCE_LENGTH, int) or STAGE1_SEQUENCE_LENGTH <= 0:
        raise ValueError("STAGE1_SEQUENCE_LENGTH 必須是正整數")
    print(f"  ✓ STAGE1_SEQUENCE_LENGTH: {STAGE1_SEQUENCE_LENGTH}")
    
    # Check time_series_split
    if time_series_split is None:
        raise ValueError("time_series_split 函數未定義。請確保已執行 Step 3")
    
    print("✓ 所有輸入驗證完成\n")
    
    return df, feature_cols, stage1_model, STAGE1_SEQUENCE_LENGTH, time_series_split


def main():
    """
    Main execution function.
    """
    print("="*80)
    print("Stage 2 Step 4 - 數據分割 + 3D 轉換 + Stage 1 過濾")
    print("="*80)
    print()
    
    # Get IPython namespace
    user_ns = get_ipython_namespace()
    
    if user_ns is None:
        raise RuntimeError(
            "此腳本必須在 IPython/Jupyter/Colab 環境中執行。\n"
            "請在 Colab Cell 中執行：\n"
            "  !curl -s https://raw.githubusercontent.com/caizongxun/crypto-zigzag-ml/main/scripts/stage2_step4_runner.py | python3"
        )
    
    # Validate inputs
    df, feature_cols, stage1_model, STAGE1_SEQUENCE_LENGTH, time_series_split = validate_inputs(user_ns)
    
    # Step 4A: Split data
    print("[4A/4D] 分割數據...")
    train_df, val_df, test_df = time_series_split(df, train_ratio=0.7, validation_ratio=0.15)
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Val: {len(val_df):,} rows")
    print(f"  Test: {len(test_df):,} rows")
    print()
    
    # Extract 2D features
    X_train_2d = train_df[feature_cols].values
    y_train_2d = train_df['zigzag_label'].values
    X_val_2d = val_df[feature_cols].values
    y_val_2d = val_df['zigzag_label'].values
    X_test_2d = test_df[feature_cols].values
    y_test_2d = test_df['zigzag_label'].values
    
    print("2D 特徵形狀：")
    print(f"  X_train_2d: {X_train_2d.shape}")
    print(f"  X_val_2d: {X_val_2d.shape}")
    print(f"  X_test_2d: {X_test_2d.shape}")
    print()
    
    # Step 4B: Convert to 3D sequences
    print(f"[4B/4D] 轉換為 3D 序列 (seq_length={STAGE1_SEQUENCE_LENGTH})...")
    X_train_3d, y_train_3d = create_3d_sequences(X_train_2d, y_train_2d, seq_length=STAGE1_SEQUENCE_LENGTH)
    X_val_3d, y_val_3d = create_3d_sequences(X_val_2d, y_val_2d, seq_length=STAGE1_SEQUENCE_LENGTH)
    X_test_3d, y_test_3d = create_3d_sequences(X_test_2d, y_test_2d, seq_length=STAGE1_SEQUENCE_LENGTH)
    
    print("3D 序列形狀：")
    print(f"  X_train_3d: {X_train_3d.shape}")
    print(f"  X_val_3d: {X_val_3d.shape}")
    print(f"  X_test_3d: {X_test_3d.shape}")
    
    # Verify shape compatibility
    model_input_shape = stage1_model.input_shape
    if X_train_3d.shape[1] != model_input_shape[1] or X_train_3d.shape[2] != model_input_shape[2]:
        raise ValueError(
            f"形狀不匹配！模型期望 (seq={model_input_shape[1]}, "
            f"features={model_input_shape[2]})，"
            f"得到 (seq={X_train_3d.shape[1]}, features={X_train_3d.shape[2]})"
        )
    print("✓ 形狀驗證通過")
    print()
    
    # Step 4C: Apply Stage 1 Model
    print("[4C/4D] 應用 Stage 1 模型...")
    
    # Train set
    print("  === 訓練集 ===")
    stage1_probs_train = stage1_model.predict(X_train_3d, batch_size=32, verbose=0)
    stage1_preds_train = (stage1_probs_train[:, 1] > 0.5).astype(int)
    signal_mask = stage1_preds_train == 1
    print(f"    信號檢測: {signal_mask.sum()} / {len(X_train_3d)} ({100*signal_mask.sum()/len(X_train_3d):.2f}%)")
    
    X_stage2_train_3d = X_train_3d[signal_mask]
    y_stage2_train = y_train_3d[signal_mask]
    X_stage2_train = X_stage2_train_3d[:, -1, :]  # 取最後一個時間步
    
    valid_mask = y_stage2_train > 0
    X_stage2_train = X_stage2_train[valid_mask]
    y_stage2_train = y_stage2_train[valid_mask]
    print(f"    有效 Stage 2 樣本: {len(X_stage2_train):,}")
    print(f"    X_stage2_train shape: {X_stage2_train.shape}")
    
    # Validation set
    print()
    print("  === 驗證集 ===")
    stage1_probs_val = stage1_model.predict(X_val_3d, batch_size=32, verbose=0)
    stage1_preds_val = (stage1_probs_val[:, 1] > 0.5).astype(int)
    signal_mask_val = stage1_preds_val == 1
    print(f"    信號檢測: {signal_mask_val.sum()} / {len(X_val_3d)} ({100*signal_mask_val.sum()/len(X_val_3d):.2f}%)")
    
    X_stage2_val_3d = X_val_3d[signal_mask_val]
    y_stage2_val = y_val_3d[signal_mask_val]
    X_stage2_val = X_stage2_val_3d[:, -1, :]
    
    valid_mask_val = y_stage2_val > 0
    X_stage2_val = X_stage2_val[valid_mask_val]
    y_stage2_val = y_stage2_val[valid_mask_val]
    print(f"    有效 Stage 2 樣本: {len(X_stage2_val):,}")
    print(f"    X_stage2_val shape: {X_stage2_val.shape}")
    
    # Test set
    print()
    print("  === 測試集 ===")
    stage1_probs_test = stage1_model.predict(X_test_3d, batch_size=32, verbose=0)
    stage1_preds_test = (stage1_probs_test[:, 1] > 0.5).astype(int)
    signal_mask_test = stage1_preds_test == 1
    print(f"    信號檢測: {signal_mask_test.sum()} / {len(X_test_3d)} ({100*signal_mask_test.sum()/len(X_test_3d):.2f}%)")
    
    X_stage2_test_3d = X_test_3d[signal_mask_test]
    y_stage2_test = y_test_3d[signal_mask_test]
    X_stage2_test = X_stage2_test_3d[:, -1, :]
    
    valid_mask_test = y_stage2_test > 0
    X_stage2_test = X_stage2_test[valid_mask_test]
    y_stage2_test = y_stage2_test[valid_mask_test]
    print(f"    有效 Stage 2 樣本: {len(X_stage2_test):,}")
    print(f"    X_stage2_test shape: {X_stage2_test.shape}")
    print()
    
    print("✓ Stage 1 過濾完成")
    print()
    
    # Step 4D: Save to global namespace
    print("[4D/4D] 將結果保存到全局命名空間...")
    user_ns['X_stage2_train'] = X_stage2_train
    user_ns['y_stage2_train'] = y_stage2_train
    user_ns['X_stage2_val'] = X_stage2_val
    user_ns['y_stage2_val'] = y_stage2_val
    user_ns['X_stage2_test'] = X_stage2_test
    user_ns['y_stage2_test'] = y_stage2_test
    
    # Optionally save to disk
    STAGE2_DATA_DIR = user_ns.get('STAGE2_DATA_DIR')
    if STAGE2_DATA_DIR:
        STAGE2_DATA_DIR = Path(STAGE2_DATA_DIR)
        STAGE2_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(STAGE2_DATA_DIR / 'X_stage2_train.pkl', 'wb') as f:
            pickle.dump(X_stage2_train, f)
        with open(STAGE2_DATA_DIR / 'y_stage2_train.pkl', 'wb') as f:
            pickle.dump(y_stage2_train, f)
        with open(STAGE2_DATA_DIR / 'X_stage2_val.pkl', 'wb') as f:
            pickle.dump(X_stage2_val, f)
        with open(STAGE2_DATA_DIR / 'y_stage2_val.pkl', 'wb') as f:
            pickle.dump(y_stage2_val, f)
        with open(STAGE2_DATA_DIR / 'X_stage2_test.pkl', 'wb') as f:
            pickle.dump(X_stage2_test, f)
        with open(STAGE2_DATA_DIR / 'y_stage2_test.pkl', 'wb') as f:
            pickle.dump(y_stage2_test, f)
        print(f"  數據已保存到: {STAGE2_DATA_DIR}")
    
    print("✓ 完成")
    print()
    
    # Summary
    print("="*80)
    print("STEP 4 完成總結")
    print("="*80)
    print(f"訓練集: X_stage2_train {X_stage2_train.shape}")
    print(f"驗證集: X_stage2_val {X_stage2_val.shape}")
    print(f"測試集: X_stage2_test {X_stage2_test.shape}")
    print()
    print("現在可以執行 Step 5 (保存數據) 或 Step 6 (訓練 Stage 2 模型)")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
