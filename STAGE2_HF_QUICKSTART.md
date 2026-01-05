# Stage 2 Training Quick Start - With HuggingFace Models

完整流程：從 HuggingFace 下載 Stage 1 模型 → 訓練 Stage 2 → 評估

總耗時：**~8 分鐘（含下載時間）**

---

## 前置條件

```bash
pip install huggingface-hub
```

---

## 快速開始（3 步）

### Step 1: 打開 Notebook

```bash
cd crypto-zigzag-ml
jupyter notebook notebooks/07_stage2_hf_download_and_train.ipynb
```

### Step 2: 依次運行所有 Cell

| 步驟 | 內容 | 說明 |
|-----|------|-------|
| **Step 0** | Setup & Configuration | 設置路徑和變量 |
| **Step 1** | Download Stage 1 from HuggingFace | 從 HF 自動下載分類器 |
| **Step 2** | Load and Verify Stage 1 Model | 驗證模型載入成功 |
| **Step 3** | Download Training Data from HuggingFace | 下載 BTCUSDT 15m K 線數據 |
| **Step 4** | Feature Engineering & Labeling | 計算 ZigZag 標籤 |
| **Step 5** | Train-Val-Test Split & Stage 1 Filtering | 篩選 Stage 1 信號 |
| **Step 6** | Save Stage 2 Training Data | 保存訓練數據 |
| **Step 7** | Train Stage 2 Model | 訓練 LightGBM 分類器 |
| **Step 8** | Evaluate on Test Set | 評估性能 |
| **Step 9** | Cross-Validation | 5 折交叉驗證 |
| **Step 10** | Summary Report | 生成總結報告 |

### Step 3: 查看結果

所有 Notebook cell 都有 print 輸出，包括：
- 下載進度
- 模型摘要
- 數據統計
- 訓練精度
- 測試性能
- 交叉驗證結果

---

## HuggingFace 模型結構

### 下載位置

```
https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/tree/main/v1_model/
├── BTCUSDT/
│   ├── 15m/
│   │   ├── classification.h5      (1.56 MB)
│   │   └── params.json
│   └── 1h/
│       ├── classification.h5
│       └── params.json
├── ETHUSDT/
│   ├── 15m/...
│   └── 1h/...
└── ... (其他 20 個幣種)
```

### 自動下載代碼

```python
from huggingface_hub import hf_hub_download
from pathlib import Path

HF_DATASET_ID = 'zongowo111/v2-crypto-ohlcv-data'
HF_MODEL_PATH = 'v1_model/BTCUSDT/15m'
LOCAL_DIR = Path('models/stage1/btcusdt_15m')

# 下載 classification.h5
classification_path = hf_hub_download(
    repo_id=HF_DATASET_ID,
    filename=f'{HF_MODEL_PATH}/classification.h5',
    repo_type='dataset',
    cache_dir=str(LOCAL_DIR)
)

# 下載 params.json
params_path = hf_hub_download(
    repo_id=HF_DATASET_ID,
    filename=f'{HF_MODEL_PATH}/params.json',
    repo_type='dataset',
    cache_dir=str(LOCAL_DIR)
)

print(f'Downloaded: {classification_path}')
```

---

## 預期輸出

### 下載進度
```
Downloading Stage 1 model from HuggingFace...
Dataset: zongowo111/v2-crypto-ohlcv-data
Path: v1_model/BTCUSDT/15m
✓ Downloaded classification.h5: .../classification.h5
✓ Downloaded params.json: .../params.json
```

### 模型驗證
```
Loading Stage 1 model from: .../classification.h5
✓ Model loaded successfully

Model Summary:
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
Total params: XXX,XXX
Trainable params: XXX,XXX
Non-trainable params: 0
_________________________________________________________________

Test prediction shape: (1, 2)
Test prediction value: [0.xx 0.xx]
```

### 數據統計
```
=== TRAIN SET ===
Applying Stage 1 model to train set...
Signals detected: 4,500 / 150,000 (3.00%)
Valid Stage 2 samples (train): 3,200
Label distribution (train):
  Label 1: 750 (23.4%)
  Label 2: 850 (26.6%)
  Label 3: 900 (28.1%)
  Label 4: 700 (21.9%)

=== VALIDATION SET ===
Signals detected: 1,080 / 32,000 (3.38%)
Valid Stage 2 samples (val): 800

=== TEST SET ===
Signals detected: 540 / 16,000 (3.38%)
Valid Stage 2 samples (test): 350
```

### 訓練結果
```
Training Stage 2 model...
Train samples: 3,200
Val samples: 800
Test samples: 350

Training Results:
  train_accuracy: 0.8950
  val_accuracy: 0.8620

Test Metrics:
  accuracy: 0.8543
  f1_score: 0.8521
  precision: 0.8634
  recall: 0.8417

Cross-Validation Results:
  Mean Accuracy: 0.8512
  Std Accuracy: 0.0187
  Min Accuracy: 0.8234
  Max Accuracy: 0.8821
```

### 最終摘要
```
================================================================================
STAGE 2 TRAINING SUMMARY - BTCUSDT 15m
================================================================================

Data Preparation:
  Original K-bars: 198,000
  Stage 1 Signals: 6,120 (3.09%)
  Stage 2 Valid: 4,350

Train/Val/Test Split:
  Train: 3,200
  Val: 800
  Test: 350

Model Performance:
  Train Accuracy: 0.8950
  Val Accuracy: 0.8620
  Test Accuracy: 0.8543
  Test F1-Score: 0.8521

Cross-Validation:
  Mean Accuracy: 0.8512 +/- 0.0187

Models Saved:
  Location: models/stage2/btcusdt_15m
    - btcusdt_15m_stage2_model.txt (500 KB)
    - btcusdt_15m_stage2_scaler.pkl (50 KB)
================================================================================
```

---

## 訓練其他幣種

### 修改 Notebook 變數

在 **Step 0** 中修改：

```python
SYMBOL = 'ETHUSDT'  # 改為 ETHUSDT
TIMEFRAME = '1h'    # 改為 1h
```

然後運行所有 cell。系統會自動：
1. 從 HF 下載對應的 Stage 1 模型
2. 下載對應的 K 線數據
3. 訓練新的 Stage 2 模型

### 支持的幣種和時間框架

所有 22 個幣種都有 15m 和 1h 的 Stage 1 模型：

```
BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, SOLUSDT, XRP USDT,
DOGEUSDT, DOTUSDT, AVAXUSDT, POLYUSDT, LINKUSDT, LTCUSDT,
BCHUSDT, UNIUSDT, ATOMUSDT, ALGOUSDT, FILUSDT, OPUSDT,
ARBUSDT, MATICUSDT, AAVEUSDT, NEARUSDT
```

---

## 批量訓練所有 22 個幣種

### 方式 1：使用批量訓練腳本

```bash
python scripts/train_all_stage2.py
```

（需要先創建此腳本）

### 方式 2：迴圈運行 Notebook

```python
import subprocess

symbols_timeframes = [
    ('BTCUSDT', '15m'), ('BTCUSDT', '1h'),
    ('ETHUSDT', '15m'), ('ETHUSDT', '1h'),
    # ... 更多
]

for symbol, timeframe in symbols_timeframes:
    # 修改 Notebook 的 Step 0
    # 運行所有 cell
    # 保存結果
    pass
```

---

## 常見問題

### 問題 1：下載速度慢
**解決方案：**
- HuggingFace 在首次下載時需要時間
- 文件會被緩存，後續調用會更快
- 可以同時下載多個幣種的數據

### 問題 2：Stage 1 模型找不到
**解決方案：**
```python
# 檢查 HuggingFace 上是否有這個文件
from huggingface_hub import list_files_in_repo

files = list_files_in_repo(
    repo_id='zongowo111/v2-crypto-ohlcv-data',
    repo_type='dataset'
)
print([f for f in files if 'v1_model' in f])
```

### 問題 3：Stage 2 精度很低
**檢查清單：**
1. 確認 Stage 1 信號篩選正確（3-5% 通過率）
2. 檢查 ZigZag 標籤分佈（應該相對均勻）
3. 增加 LightGBM 的 `num_boost_round` 參數
4. 檢查特徵工程是否正確

### 問題 4：內存不足
**解決方案：**
```python
# 減少批次大小或分割數據
X_train = X_stage2_train[::2]  # 每隔一個樣本採樣
y_train = y_stage2_train[::2]
```

---

## 下一步

完成 Stage 2 訓練後：

1. **推理演示**
   - 使用 `notebooks/06_stage2_inference_demo.ipynb`
   - 或 `src/stage2_inference.py` 直接推理

2. **模型上傳**
   - 將訓練好的 Stage 2 模型上傳到 HuggingFace
   - 便於版本管理和共享

3. **實時交易**
   - 集成到交易機器人
   - 使用完整的二階段管道

---

## 技術細節

### 自動化下載機制

Notebook 使用 `huggingface_hub` 庫自動：

1. **檢查本地緩存** - 如果已下載，直接使用
2. **驗證完整性** - 檢查文件哈希
3. **智能重試** - 下載失敗自動重試
4. **進度顯示** - 實時顯示下載進度

### 模型配置

Stage 2 使用的配置：

```python
# LightGBM 參數
training_params = {
    'objective': 'multiclass',
    'num_class': 4,
    'num_boost_round': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'num_leaves': 31,
}

# 驗證參數
validation_params = {
    'metric': 'multi_error',
    'early_stopping_rounds': 20,
}
```

---

## 文件結構

訓練完成後的本地結構：

```
crypto-zigzag-ml/
├── models/
│   ├── stage1/
│   │   └── btcusdt_15m/
│   │       ├── classification.h5  (from HF)
│   │       └── params.json
│   └── stage2/
│       └── btcusdt_15m/
│           ├── btcusdt_15m_stage2_model.txt
│           └── btcusdt_15m_stage2_scaler.pkl
└── data/
    ├── stage2/
    │   └── btcusdt_15m/
    │       ├── X_stage2_train.pkl
    │       ├── y_stage2_train.pkl
    │       ├── X_stage2_val.pkl
    │       ├── y_stage2_val.pkl
    │       ├── X_stage2_test.pkl
    │       └── y_stage2_test.pkl
```

---

**準備好了？打開 Jupyter 開始訓練！**

```bash
jupyter notebook notebooks/07_stage2_hf_download_and_train.ipynb
```
