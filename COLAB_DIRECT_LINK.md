# 🚀 一鍵在 Google Colab 啟動 Stage 2 訓練

## 最快方式 (30 秒開始訓練)

### 點這裡直接在 Colab 打開 👇

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/caizongxun/crypto-zigzag-ml/blob/main/notebooks/08_stage2_colab_training.ipynb)

---

## 手動方式

1. 打開 Google Colab
   ```
   https://colab.research.google.com/
   ```

2. 選擇 **File** → **Open notebook**

3. 貼上以下連結
   ```
   https://colab.research.google.com/github/caizongxun/crypto-zigzag-ml/blob/main/notebooks/08_stage2_colab_training.ipynb
   ```

4. 按 **Enter**

---

## 設置 GPU (可選，但推薦)

1. 點擊 **Runtime** 選單
2. 選擇 **Change runtime type**
3. 選擇 **GPU** (T4 或 V100)
4. 點擊 **Save**

### GPU 性能提升
- **無 GPU**: ~2 分鐘
- **T4 GPU**: ~30 秒 (4x 更快)
- **V100 GPU**: ~15 秒 (8x 更快)

---

## 快速訓練 (全自動，3 步)

### Step 1: 點擊上面的 Colab 按鈕

### Step 2: 等待頁面加載

### Step 3: 點擊 ▶️ (播放按鈕)

**就是這樣！**

---

## 訓練流程自動執行 (5-8 分鐘)

```
1. 環境安裝 (1 分鐘)
   ✓ 安裝必要套件
   ✓ 克隆 GitHub 倉庫
   ✓ 檢查 GPU

2. 下載模型和數據 (2 分鐘)
   ✓ 從 HuggingFace 下載 Stage 1 模型
   ✓ 從 HuggingFace 下載 K 線數據

3. 特徵工程 (2 分鐘)
   ✓ 計算 ZigZag 標籤
   ✓ 計算技術指標
   ✓ 篩選 Stage 1 信號

4. 訓練 Stage 2 (1-2 分鐘)
   ✓ 訓練 LightGBM 分類器
   ✓ 5 折交叉驗證
   ✓ 測試集評估

5. 保存結果 (自動)
   ✓ 模型保存到 Google Drive
   ✓ 訓練數據保存到 Google Drive
```

---

## 預期結果

訓練完成後你會看到：

```
================================================================================
STAGE 2 TRAINING COMPLETE - BTCUSDT 15m
================================================================================

DATA STATISTICS:
  Original K-bars: 198,000
  Stage 1 Signals: 6,120
  Stage 2 Valid Samples: 4,350

TRAIN/VAL/TEST SPLIT:
  Train: 3,200
  Val: 800
  Test: 350

MODEL PERFORMANCE:
  Train Accuracy: 0.8950
  Val Accuracy: 0.8620
  Test Accuracy: 0.8543
  Test F1-Score: 0.8521

CROSS-VALIDATION:
  Mean Accuracy: 0.8512 +/- 0.0187

MODELS SAVED:
  btcusdt_15m_stage2_model.txt (500 KB)
  btcusdt_15m_stage2_scaler.pkl (50 KB)

================================================================================
```

---

## 訓練後下載結果

所有檔案自動保存到：
```
Google Drive / Colab Results / Stage2 / btcusdt_15m /
```

包括：
- `btcusdt_15m_stage2_model.txt` - 訓練好的模型
- `btcusdt_15m_stage2_scaler.pkl` - 特徵正規化器
- `training_data/` - 訓練數據

---

## 注意事項

⚠️ **Colab 單次執行時間限制**
- 免費版本: 12 小時連續執行
- Pro 版本: 24 小時連續執行
- 我們的訓練只需 5-8 分鐘，完全沒問題

⚠️ **檔案會在 Colab 中保留多久**
- 訓練完成後自動保存到 Google Drive
- Colab 本身的檔案在 12 小時後清除
- **你的 Google Drive 中的檔案永久保存**

---

## 訓練其他幣種

1. 複製上面的 Notebook 連結
2. 在 Colab 中修改變數
   ```python
   SYMBOL = 'ETHUSDT'   # 改成其他幣種
   TIMEFRAME = '1h'     # 改成其他時間框架
   ```
3. 重新執行所有 cell

---

## 問題排查

### 問題 1: "Notebook not found"
- 檢查 URL 是否正確
- 或使用上面的 [![Open In Colab] 按鈕直接打開

### 問題 2: "GPU timeout"
- 很少發生，如果發生就重新執行 cell
- Colab 會自動重連

### 問題 3: "下載太慢"
- 這是 HuggingFace 的速度限制
- 第一次下載會比較慢，後續會使用快取

### 問題 4: "沒看到 GPU"
```python
# 在任何 cell 執行：
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## 下一步

✓ **Stage 2 訓練完成**

⏭️ **接下來可以：**
1. 訓練其他 21 個幣種 (使用迴圈或批量腳本)
2. 執行推理演示 (`notebooks/06_stage2_inference_demo.ipynb`)
3. 部署到生產環境 (Flask API)

---

**準備好了嗎？ 🚀**

[按這裡在 Colab 開始訓練](https://colab.research.google.com/github/caizongxun/crypto-zigzag-ml/blob/main/notebooks/08_stage2_colab_training.ipynb)
