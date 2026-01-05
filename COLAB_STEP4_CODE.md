# Colab Stage 2 Step 4 - 完整代碼

在 Colab Cell 中複製貼上下方代碼並執行（Shift+Enter）

```python
# Step 4: Create 3D Time Series Features & Apply Stage 1 Model
print(f'[4/7] Data splitting, 3D conversion, and Stage 1 filtering...')

def create_3d_sequences(X, y, seq_length=10):
    """Convert 2D features to 3D sequences for LSTM/CNN models"""
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int32)

# First, split data into train/val/test
print(f'Splitting data...')
train_df, val_df, test_df = time_series_split(df, train_ratio=0.7, validation_ratio=0.15)

# Convert to 2D arrays
X_train_2d = train_df[feature_cols].values
y_train_2d = train_df['zigzag_label'].values
X_val_2d = val_df[feature_cols].values
y_val_2d = val_df['zigzag_label'].values
X_test_2d = test_df[feature_cols].values
y_test_2d = test_df['zigzag_label'].values

print(f'Converting to 3D sequences (seq_length={STAGE1_SEQUENCE_LENGTH})...')

# Create 3D sequences
X_train_3d, y_train_3d = create_3d_sequences(X_train_2d, y_train_2d, seq_length=STAGE1_SEQUENCE_LENGTH)
X_val_3d, y_val_3d = create_3d_sequences(X_val_2d, y_val_2d, seq_length=STAGE1_SEQUENCE_LENGTH)
X_test_3d, y_test_3d = create_3d_sequences(X_test_2d, y_test_2d, seq_length=STAGE1_SEQUENCE_LENGTH)

print(f'  Train: {X_train_3d.shape}')
print(f'  Val: {X_val_3d.shape}')
print(f'  Test: {X_test_3d.shape}')
print(f'✓ Conversion complete')

# === Apply Stage 1 Model ===
print(f'Applying Stage 1 model...')

# Train set
stage1_probs_train = stage1_model.predict(X_train_3d, verbose=0)
stage1_preds_train = (stage1_probs_train[:, 1] > 0.5).astype(int)
signal_mask = stage1_preds_train == 1
X_stage2_train_3d = X_train_3d[signal_mask]
y_stage2_train = y_train_3d[signal_mask]
X_stage2_train = X_stage2_train_3d[:, -1, :]  # Use last timestep
valid_mask = y_stage2_train > 0
X_stage2_train = X_stage2_train[valid_mask]
y_stage2_train = y_stage2_train[valid_mask]
print(f'  Train: {len(X_stage2_train):,} samples')

# Validation set
stage1_probs_val = stage1_model.predict(X_val_3d, verbose=0)
stage1_preds_val = (stage1_probs_val[:, 1] > 0.5).astype(int)
signal_mask_val = stage1_preds_val == 1
X_stage2_val_3d = X_val_3d[signal_mask_val]
y_stage2_val = y_val_3d[signal_mask_val]
X_stage2_val = X_stage2_val_3d[:, -1, :]
valid_mask_val = y_stage2_val > 0
X_stage2_val = X_stage2_val[valid_mask_val]
y_stage2_val = y_stage2_val[valid_mask_val]
print(f'  Val: {len(X_stage2_val):,} samples')

# Test set
stage1_probs_test = stage1_model.predict(X_test_3d, verbose=0)
stage1_preds_test = (stage1_probs_test[:, 1] > 0.5).astype(int)
signal_mask_test = stage1_preds_test == 1
X_stage2_test_3d = X_test_3d[signal_mask_test]
y_stage2_test = y_test_3d[signal_mask_test]
X_stage2_test = X_stage2_test_3d[:, -1, :]
valid_mask_test = y_stage2_test > 0
X_stage2_test = X_stage2_test[valid_mask_test]
y_stage2_test = y_stage2_test[valid_mask_test]
print(f'  Test: {len(X_stage2_test):,} samples')

print(f'✓ Stage 1 filtering complete')
```

## 說明

- **輸入**: 使用 Step 1-3 中定義的 `df`, `feature_cols`, `stage1_model`, `time_series_split`, `STAGE1_SEQUENCE_LENGTH`
- **輸出**: 生成以下變數供 Step 5+ 使用
  - `X_stage2_train`, `y_stage2_train`
  - `X_stage2_val`, `y_stage2_val`
  - `X_stage2_test`, `y_stage2_test`

## 預期輸出

```
[4/7] Data splitting, 3D conversion, and Stage 1 filtering...
Splitting data...
Converting to 3D sequences (seq_length=10)...
  Train: (6991, 10, 84)
  Val: (1491, 10, 84)
  Test: (1491, 10, 84)
✓ Conversion complete
Applying Stage 1 model...
  Train: 500 samples
  Val: 100 samples
  Test: 100 samples
✓ Stage 1 filtering complete
```
