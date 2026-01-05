# Colab Stage 2 訓練快速啟動指南

## 🚀 超快速執行（2 行代碼搞定）

在 Colab 中的任何 Cell 執行以下代碼，將自動完成 **Step 1-4 的所有流程**：

```python
!curl -s https://raw.githubusercontent.com/caizongxun/crypto-zigzag-ml/main/scripts/stage2_step4_runner.py | python3
```

或者使用 `wget`：

```python
!wget -q https://raw.githubusercontent.com/caizongxun/crypto-zigzag-ml/main/scripts/stage2_step4_runner.py -O /tmp/runner.py && python3 /tmp/runner.py
```

---

## 📋 完整流程（假設你已經跑過 Step 1-3）

### 前置條件

在執行 `stage2_step4_runner.py` 之前，確保已經執行過：

1. **Step 1**: 下載 Stage 1 模型 ✓
   ```python
   # 應該有以下變數定義
   stage1_model  # Keras model (已加載)
   STAGE1_SEQUENCE_LENGTH  # = 10
   STAGE1_MODEL_DIR  # Path object
   ```

2. **Step 2**: 下載訓練數據 ✓
   ```python
   # 應該有
   df  # DataFrame with raw OHLCV data
   data_file  # Path to parquet file
   ```

3. **Step 3**: 特徵工程 ✓
   ```python
   # 應該有
   df  # DataFrame (已加上 zigzag_label 和技術指標)
   feature_cols  # list of feature column names
   ```

### 執行 Step 4 (新方式)

**只需一行代碼：**

```python
!curl -s https://raw.githubusercontent.com/caizongxun/crypto-zigzag-ml/main/scripts/stage2_step4_runner.py | python3
```

**預期輸出：**

```
================================================================================
Stage 2 Step 4 - 數據分割 + 3D 轉換 + Stage 1 過濾
================================================================================
驗證輸入...
  ✓ df shape: (10000, 120)
  ✓ feature_cols: 84 features
  ✓ stage1_model input shape: (None, 10, 20)
  ✓ STAGE1_SEQUENCE_LENGTH: 10
✓ 所有輸入驗證完成

[4A/4D] 分割數據...
  Train: 7,000 rows
  Val: 1,500 rows
  Test: 1,500 rows

[4B/4D] 轉換為 3D 序列 (seq_length=10)...
3D 序列形狀：
  X_train_3d: (6991, 10, 84)
  X_val_3d: (1491, 10, 84)
  X_test_3d: (1491, 10, 84)
✓ 形狀驗證通過

[4C/4D] 應用 Stage 1 模型...
  === 訓練集 ===
    信號檢測: 699 / 6991 (10.00%)
    有效 Stage 2 樣本: 500
    X_stage2_train shape: (500, 84)

  === 驗證集 ===
    信號檢測: 150 / 1491 (10.06%)
    有效 Stage 2 樣本: 100
    X_stage2_val shape: (100, 84)

  === 測試集 ===
    信號檢測: 150 / 1491 (10.06%)
    有效 Stage 2 樣本: 100
    X_stage2_test shape: (100, 84)

✓ Stage 1 過濾完成

[4D/4D] 將結果保存到全局命名空間...
  數據已保存到: data/stage2/btcusdt_15m

✓ 完成

================================================================================
STEP 4 完成總結
================================================================================
訓練集: X_stage2_train (500, 84)
驗證集: X_stage2_val (100, 84)
測試集: X_stage2_test (100, 84)

現在可以執行 Step 5 (保存數據) 或 Step 6 (訓練 Stage 2 模型)
================================================================================
```

執行完後，以下變數會自動在 Colab 命名空間中定義：

```python
X_stage2_train  # shape: (n, 84)
y_stage2_train  # shape: (n,)
X_stage2_val    # shape: (m, 84)
y_stage2_val    # shape: (m,)
X_stage2_test   # shape: (k, 84)
y_stage2_test   # shape: (k,)
```

---

## 🔄 接下來的步驟

### Step 5: 保存 Stage 2 訓練數據

```python
print('[5/7] Saving Stage 2 training data...')

with open(STAGE2_DATA_DIR / 'X_stage2_train.pkl', 'wb') as f:
    pickle.dump(X_stage2_train, f)
with open(STAGE2_DATA_DIR / 'y_stage2_train.pkl', 'wb') as f:
    pickle.dump(y_stage2_train, f)
# ... 類似地保存 val 和 test

print('✓ Data saved')
```

### Step 6: 訓練 Stage 2 模型

```python
from src.stage2_trainer import Stage2Trainer

print('[6/7] Training Stage 2 model...')
trainer = Stage2Trainer(model_dir=str(STAGE2_MODEL_DIR))

train_results = trainer.train(
    X_stage2_train, y_stage2_train,
    X_stage2_val, y_stage2_val,
    normalize=True,
    cv_folds=5,
    save_model=True
)

print('✓ Model trained')
```

### Step 7: 評估模型

```python
print('[7/7] Evaluation and cross-validation...')

test_metrics = trainer.evaluate(X_stage2_test, y_stage2_test)
print(f'Test Accuracy: {test_metrics["accuracy"]:.4f}')
print(f'Test F1-Score: {test_metrics["f1_score"]:.4f}')
```

---

## 🆘 故障排除

### 問題 1: 找不到變數

**症狀：**
```
ValueError: df 必須是 DataFrame
```

**解決方案：**
確保已執行 Step 1-3，並且 `df`、`feature_cols`、`stage1_model` 都已定義。在執行 runner 前添加檢查：

```python
print(f"df: {df.shape if 'df' in dir() else 'NOT DEFINED'}")
print(f"feature_cols: {len(feature_cols) if 'feature_cols' in dir() else 'NOT DEFINED'}")
print(f"stage1_model: {'OK' if 'stage1_model' in dir() else 'NOT DEFINED'}")
```

### 問題 2: 記憶體不足

**症狀：**
```
MemoryError during prediction
```

**解決方案：**
修改 runner 中的 batch_size（第 131 行）：

```python
# 改成更小的 batch_size
stage1_probs_train = stage1_model.predict(X_train_3d, batch_size=16, verbose=0)
```

### 問題 3: 形狀不匹配

**症狀：**
```
ValueError: 形狀不匹配！模型期望 (seq=10, features=20), 得到 (seq=10, features=84)
```

**解決方案：**
Runner 會自動檢查形狀。如果仍然出錯，確認 Stage 1 模型確實需要 (10, 20) 輸入：

```python
print(stage1_model.input_shape)
# 應該打印: (None, 10, 20)
```

---

## 📊 輸出文件

執行完 Step 4 後，以下文件會被自動保存（如果 `STAGE2_DATA_DIR` 已定義）：

```
data/stage2/btcusdt_15m/
├── X_stage2_train.pkl    # 訓練集特徵 (n, 84)
├── y_stage2_train.pkl    # 訓練集標籤 (n,)
├── X_stage2_val.pkl      # 驗證集特徵 (m, 84)
├── y_stage2_val.pkl      # 驗證集標籤 (m,)
├── X_stage2_test.pkl     # 測試集特徵 (k, 84)
└── y_stage2_test.pkl     # 測試集標籤 (k,)
```

---

## ✅ 完整 Colab 代碼示例

假設你已經完成 Step 1-3，這是完整的 Stage 2 流程：

```python
# ============= Step 4: 執行 Runner =============
!curl -s https://raw.githubusercontent.com/caizongxun/crypto-zigzag-ml/main/scripts/stage2_step4_runner.py | python3

# ============= Step 5: 保存數據 =============
print('[5/7] Saving Stage 2 training data...')
import pickle

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

print('✓ Data saved')

# ============= Step 6: 訓練 Stage 2 =============
from src.stage2_trainer import Stage2Trainer

print('[6/7] Training Stage 2 model...')
trainer = Stage2Trainer(model_dir=str(STAGE2_MODEL_DIR))

train_results = trainer.train(
    X_stage2_train, y_stage2_train,
    X_stage2_val, y_stage2_val,
    normalize=True,
    cv_folds=5,
    save_model=True
)

print(f'Train Accuracy: {train_results["train_accuracy"]:.4f}')
print(f'Val Accuracy: {train_results["val_accuracy"]:.4f}')

# ============= Step 7: 評估 =============
print('[7/7] Evaluation...')
test_metrics = trainer.evaluate(X_stage2_test, y_stage2_test)
print(f'Test Accuracy: {test_metrics["accuracy"]:.4f}')
print(f'Test F1-Score: {test_metrics["f1_score"]:.4f}')
```

---

## 💡 Tip

如果你多次執行，可以把 Step 4 的執行命令保存為一個可重用的函數：

```python
def run_stage2_step4():
    """在 Colab 中執行 Stage 2 Step 4"""
    import subprocess
    result = subprocess.run(
        'curl -s https://raw.githubusercontent.com/caizongxun/crypto-zigzag-ml/main/scripts/stage2_step4_runner.py | python3',
        shell=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode == 0

# 使用
if run_stage2_step4():
    print("✓ Step 4 完成！")
else:
    print("✗ Step 4 失敗")
```

---

**更多問題？** 查看 `troubleshooting_summary.md`
