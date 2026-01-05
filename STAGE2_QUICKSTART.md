# Stage 2 Quick Start (5 Minutes)

Fast track to train and deploy Stage 2 model.

---

## Prerequisites

✓ Stage 1 model trained (`models/btcusdt_15m/classification.h5`)
✓ Training data prepared (`data/raw/BTC_15m.parquet`)
✓ Python 3.8+ with dependencies installed

---

## 3-Step Training

### Step 1: Prepare Data (2 minutes)

```bash
jupyter notebook notebooks/04_stage2_data_prep.ipynb
```

In Jupyter:
1. Run all cells (Shift+Enter repeatedly)
2. Wait for output showing label distribution
3. Verify files saved to `data/stage2/`

**Expected output:**
```
Stage 2 Training: 5000 samples
  HH: 750 (15%)
  LH: 1250 (25%)
  HL: 1750 (35%)
  LL: 1250 (25%)
```

---

### Step 2: Train Model (3 minutes)

```bash
jupyter notebook notebooks/05_stage2_training.ipynb
```

In Jupyter:
1. Run all cells
2. Monitor training progress
3. Wait for test accuracy report

**Expected metrics:**
- Test Accuracy: 0.83-0.87
- F1-Score: 0.84-0.88
- Cross-Validation: 0.85 +/- 0.03

**Model saved automatically to:**
```
models/stage2/btcusdt_15m_stage2_model.txt
models/stage2/btcusdt_15m_stage2_scaler.pkl
```

---

### Step 3: Test Inference (1 minute)

```bash
jupyter notebook notebooks/06_stage2_inference_demo.ipynb
```

In Jupyter:
1. Run all cells
2. Review sample predictions
3. Check summary statistics and confidence distribution

**Key output:**
```
Total Samples: 1000
BUY Signals: 35
SELL Signals: 28
No Signals: 937
Average Combined Confidence: 0.88
```

---

## Use in Production

### Option A: Python Script

```python
from src.stage2_inference import PipelineInference
import numpy as np

# Initialize
pipeline = PipelineInference(model_dir='models')
pipeline.load_stage1_model('btcusdt_15m')
pipeline.load_stage2_model('btcusdt_15m')

# Predict
X_new = np.random.randn(100, 86)  # Your feature matrix
predictions = pipeline.predict(X_new)

# Use results
for i in range(len(X_new)):
    if predictions['action'][i] == 'BUY':
        print(f"Buy signal at index {i}")
    elif predictions['action'][i] == 'SELL':
        print(f"Sell signal at index {i}")
```

### Option B: Real-Time on Single Bar

```python
# When new K-bar arrives
new_features = extract_features(latest_ohlcv)  # shape: (1, 86)

result = pipeline.predict_single(new_features)

if result['action'] == 'BUY':
    place_buy_order()
elif result['action'] == 'SELL':
    place_sell_order()

print(f"Confidence: {result['combined_confidence']:.2%}")
```

### Option C: Flask API

```python
from flask import Flask, request
import numpy as np
from src.stage2_inference import PipelineInference

app = Flask(__name__)
pipeline = PipelineInference(model_dir='models')
pipeline.load_stage1_model('btcusdt_15m')
pipeline.load_stage2_model('btcusdt_15m')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    
    result = pipeline.predict_single(features, return_confidence=True)
    
    return {
        'action': result['action'],
        'signal_type': result['signal_type'],
        'confidence': result['combined_confidence']
    }

if __name__ == '__main__':
    app.run(port=5000)
```

Call from anywhere:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ..., 0.8]}'
```

---

## Verify Installation

### Check Files

```bash
ls -lh models/stage2/
ls -lh data/stage2/
```

**Should show:**
```
models/stage2/:
  btcusdt_15m_stage2_model.txt  (500 KB)
  btcusdt_15m_stage2_scaler.pkl (50 KB)

data/stage2/:
  X_stage2_train.pkl
  y_stage2_train.pkl
  X_stage2_val.pkl
  y_stage2_val.pkl
  X_stage2_test.pkl
  y_stage2_test.pkl
```

### Test Imports

```python
from src.stage2_trainer import Stage2Trainer
from src.stage2_inference import PipelineInference

print("All imports successful!")
```

---

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: Stage 1 model` | Run Stage 1 training first (Notebook 03) |
| `ModuleNotFoundError: lightgbm` | `pip install lightgbm` |
| `Low accuracy (<75%)` | Increase `num_boost_round` from 200 to 400 |
| `GPU memory error` | Use CPU: `export CUDA_VISIBLE_DEVICES=""` |
| `Stage 2 model not found` | Verify Notebook 05 completed and saved model |

---

## Performance Metrics

### Expected Results

```
Stage 1 (Signal Detection)
  Accuracy: 0.95
  Sensitivity: 0.92
  Specificity: 0.97

Stage 2 (Signal Type)
  Accuracy: 0.85
  HH Precision: 0.88
  LH Precision: 0.84
  HL Precision: 0.87
  LL Precision: 0.81

Combined Pipeline
  Overall Confidence: 0.88
  BUY Signals Precision: 0.86
  SELL Signals Precision: 0.84
```

---

## Next: Multi-Symbol Training

To train all 22 symbols:

```bash
python scripts/train_all_stage2.py
```

(Script coming soon)

---

## Documentation

- Full guide: `STAGE2_TRAINING_GUIDE.md`
- API reference: See docstrings in `src/stage2_trainer.py`
- Examples: `notebooks/04_*, 05_*, 06_*`

---

**Total Time: ~6 minutes from start to production ready**

Ready? Start with `notebooks/04_stage2_data_prep.ipynb`
