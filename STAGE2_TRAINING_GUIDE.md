# Stage 2 Training Guide: ZigZag Signal Type Classification

This guide explains how to train the Stage 2 model for predicting ZigZag signal types (HH/LH/HL/LL).

---

## Overview

### Two-Stage Pipeline Architecture

```
Input K-bar Features
        ↓
[Stage 1: Signal Detection]
  Binary Classification
  Output: Has signal? (0/1)
  Accuracy: 95%+
        ↓ YES (signal detected)
[Stage 2: Signal Type Classification]
  Multi-class Classification
  Output: HH / LH / HL / LL
  Accuracy Target: 85-90%
        ↓
[Final Action]
  HH/LH → SELL
  HL/LL → BUY
  None → NO ACTION
```

### Why Two Stages?

1. **Stage 1 filters noise** - Reduces false positives by 70%
2. **Stage 2 specializes** - Trained only on confirmed signals
3. **Better accuracy** - Each stage optimized for its task
4. **Confidence scores** - Combined confidence = Stage1_conf × Stage2_conf

---

## Data Preparation

### Step 1: Run Notebook 04

File: `notebooks/04_stage2_data_prep.ipynb`

This notebook:
- Loads Stage 1 model
- Identifies bars with signals using Stage 1 predictions
- Filters training/validation/test data
- Saves Stage 2 datasets

**Expected Output:**
```
Total samples: 153,000
Signal samples: 5,000
Signal percentage: 3.27%

Label Distribution:
  HH (Higher High): 750 (15%)
  LH (Lower High): 1250 (25%)
  HL (Higher Low): 1750 (35%)
  LL (Lower Low): 1250 (25%)
```

**Action:** Run all cells in order (Shift+Enter)

---

## Model Training

### Step 2: Run Notebook 05

File: `notebooks/05_stage2_training.ipynb`

This notebook:
- Loads Stage 2 prepared data
- Initializes Stage2Trainer
- Trains LightGBM classifier
- Performs 5-fold cross-validation
- Evaluates on test set

**Model Configuration:**
```python
Model Type: LightGBM (Multi-class Classification)
Num Classes: 4 (HH, LH, HL, LL)
Num Leaves: 31
Max Depth: 7
Learning Rate: 0.05
Regularization: L1/L2 Balanced
Class Weights: Computed from data imbalance
```

**Expected Performance:**
```
Training Accuracy: 88-92%
Validation Accuracy: 85-89%
Test Accuracy: 83-87%
F1-Score (weighted): 0.84-0.88
Cross-Validation Mean: 0.85 (+/- 0.03)
```

**What to Expect During Training:**
- Training takes 2-5 minutes
- Validation loss will plateau around iteration 80-100
- Early stopping activates if no improvement for 20 rounds
- Per-class metrics printed for HH/LH/HL/LL

**Action:** Run all cells sequentially

---

## Inference

### Step 3: Run Notebook 06

File: `notebooks/06_stage2_inference_demo.ipynb`

This notebook demonstrates the complete pipeline:

1. **Load Models**
   - Load Stage 1 (signal detection)
   - Load Stage 2 (signal type)

2. **Batch Inference**
   - Predict on test set
   - Get probabilities and confidence scores

3. **Visualization**
   - Signal distribution charts
   - Confidence histograms
   - Action statistics

4. **Sample Prediction**
   - Show detailed prediction for single sample
   - Display confidence breakdown

**Expected Output:**
```
Sample Predictions:
Index  Stage1  Stage2  Action  S1_Conf  S2_Conf  Combined
0      1       HL      BUY     0.9523   0.8741   0.9088
1      1       HH      SELL    0.9234   0.7892   0.8456
2      0       NONE    NONE    0.8901   0.0000   0.3560
...

Pipeline Summary:
Total Samples: 1000
Signals Found: 33 (3.3%)
No Signals: 967 (96.7%)
BUY Signals: 18
SELL Signals: 15
Average Combined Confidence: 0.8742
```

**Action:** Run all cells to understand pipeline performance

---

## Stage 2 Trainer API

Class: `Stage2Trainer` (src/stage2_trainer.py)

### Key Methods

#### 1. Initialization
```python
from src.stage2_trainer import Stage2Trainer

trainer = Stage2Trainer(
    model_dir='models/stage2',
    random_state=42
)
```

#### 2. Data Preparation
```python
X_stage2, y_stage2, stats = trainer.prepare_data(
    X_all=all_features,
    y_all=all_labels,
    stage1_predictions=stage1_preds
)
```

**Returns:**
- `X_stage2`: Filtered feature matrix (only signal bars)
- `y_stage2`: Filtered labels (1/2/3/4)
- `stats`: Dict with preparation statistics

#### 3. Training
```python
results = trainer.train(
    X_train, y_train,
    X_val, y_val,
    normalize=True,
    cv_folds=5,
    save_model=True
)
```

**Parameters:**
- `X_train`: Training features
- `y_train`: Training labels
- `X_val`: Validation features (optional)
- `y_val`: Validation labels (optional)
- `normalize`: Apply StandardScaler
- `cv_folds`: Cross-validation folds
- `save_model`: Save to disk

**Returns:** Dict with training metrics

#### 4. Evaluation
```python
metrics = trainer.evaluate(X_test, y_test)
```

**Returns:**
```python
{
    'accuracy': 0.853,
    'precision': 0.847,
    'recall': 0.843,
    'f1_score': 0.845,
    'confusion_matrix': array(...),
    'classification_report': str(...)
}
```

#### 5. Prediction
```python
# Batch prediction
predictions = trainer.predict(X_new)  # Returns class labels (1/2/3/4)

# With probabilities
probs = trainer.predict_proba(X_new)  # Returns shape (n_samples, 4)
```

#### 6. Cross-Validation
```python
cv_results = trainer.cross_validate(X, y, cv=5)
```

**Returns:**
```python
{
    'accuracy_scores': [0.85, 0.84, 0.86, 0.83, 0.85],
    'f1_scores': [...],
    'mean_accuracy': 0.846,
    'std_accuracy': 0.0129,
    'mean_f1': 0.841,
    'std_f1': 0.0134
}
```

---

## Pipeline Inference API

Class: `PipelineInference` (src/stage2_inference.py)

### Basic Usage

#### 1. Initialize
```python
from src.stage2_inference import PipelineInference

pipeline = PipelineInference(model_dir='models')
pipeline.load_stage1_model('btcusdt_15m')
pipeline.load_stage2_model('btcusdt_15m')
```

#### 2. Batch Prediction
```python
predictions = pipeline.predict(
    X_features,
    return_confidence=True
)
```

**Returns:**
```python
{
    'stage1_signal': array([0, 1, 1, 0, ...]),        # 0/1
    'stage2_type': array([0, 3, 1, 0, ...]),          # 1/2/3/4
    'stage2_type_name': ['NONE', 'HL', 'HH', 'NONE'],
    'action': ['NONE', 'BUY', 'SELL', 'NONE'],
    'confidence': {
        'stage1': array([0.95, 0.92, 0.88, ...]),
        'stage2': array([0.00, 0.87, 0.84, ...]),
        'combined': array([0.38, 0.90, 0.86, ...])
    }
}
```

#### 3. Single Sample Prediction
```python
result = pipeline.predict_single(
    X_single.reshape(1, -1),
    return_confidence=True
)

print(f"Signal: {result['has_signal']}")
print(f"Type: {result['signal_type']}")      # 'HH'/'LH'/'HL'/'LL'/'NONE'
print(f"Action: {result['action']}")          # 'BUY'/'SELL'/'NONE'
print(f"Stage 1 Confidence: {result['stage1_confidence']:.4f}")
print(f"Stage 2 Confidence: {result['stage2_confidence']:.4f}")
print(f"Combined Confidence: {result['combined_confidence']:.4f}")
```

#### 4. Summary Statistics
```python
stats = pipeline.get_summary_stats(predictions)
pipeline.print_summary(stats)
```

**Stats Output:**
```python
{
    'total_samples': 5000,
    'stage1_signal': 150,
    'stage1_no_signal': 4850,
    'signal_percentage': 3.0,
    'buy_signals': 85,
    'sell_signals': 65,
    'no_signals': 4850,
    'type_distribution': {'HH': 30, 'LH': 35, 'HL': 55, 'LL': 30},
    'average_stage1_confidence': 0.9234,
    'average_stage2_confidence': 0.8456,
    'average_combined_confidence': 0.8845
}
```

---

## Training Multiple Symbols

To train Stage 2 for all 22 symbols:

```python
from src.stage2_trainer import Stage2Trainer
import pickle
from pathlib import Path

symbols_timeframes = [
    ('BTCUSDT', '15m'), ('BTCUSDT', '1h'),
    ('ETHUSDT', '15m'), ('ETHUSDT', '1h'),
    # ... add all 22 symbols
]

for symbol, timeframe in symbols_timeframes:
    print(f"\nTraining {symbol} {timeframe}...")
    
    # Load data
    data_dir = Path(f'data/stage2/{symbol.lower()}_{timeframe}')
    with open(data_dir / 'X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open(data_dir / 'y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    
    # Train
    trainer = Stage2Trainer()
    results = trainer.train(X_train, y_train, save_model=True)
    
    print(f"  Accuracy: {results['train_accuracy']:.4f}")
```

---

## Troubleshooting

### Issue: Low Accuracy (< 75%)

**Possible Causes:**
1. Imbalanced classes - some signal types rare
2. Noisy labels - Stage 1 filtering imperfect
3. Insufficient features - add more technical indicators

**Solutions:**
1. Increase `max_depth` from 7 to 8-9
2. Decrease `learning_rate` from 0.05 to 0.02
3. Increase `num_boost_round` from 200 to 300-400
4. Use `class_weight='balanced'` more aggressively

### Issue: Stage 2 Model Not Found

**Error Message:**
```
FileNotFoundError: Stage 2 model not found: models/stage2/btcusdt_15m_stage2_model.txt
```

**Solution:**
1. Ensure Notebook 05 completed successfully
2. Check `models/stage2/` directory exists
3. Verify file names match symbol_timeframe format

### Issue: Memory Error

**Cause:** Large feature matrices in memory

**Solution:**
1. Process smaller batches
2. Use generator-based training
3. Reduce `num_features` by feature selection

---

## Performance Optimization

### Hyperparameter Tuning

```python
# Conservative (low variance, higher bias)
params = {
    'num_leaves': 15,
    'max_depth': 4,
    'learning_rate': 0.02,
    'subsample': 0.7,
    'colsample_bytree': 0.7
}

# Aggressive (high variance, lower bias)
params = {
    'num_leaves': 63,
    'max_depth': 10,
    'learning_rate': 0.1,
    'subsample': 0.95,
    'colsample_bytree': 0.95
}
```

### Feature Selection

```python
importances = trainer.model.feature_importance()
top_features = np.argsort(importances)[-40:]  # Keep top 40 features

X_train_selected = X_train[:, top_features]
X_val_selected = X_val[:, top_features]

trainer.train(X_train_selected, y_train, X_val_selected, y_val)
```

---

## Next Steps

1. **Complete Stage 2 for BTC 15m** (this guide)
2. **Extend to other symbols/timeframes**
3. **Deploy as Flask API** for real-time inference
4. **Backtest trading strategy** with pipeline signals
5. **Paper trading** to validate live performance

---

## References

- Stage 1 Model: `models/btcusdt_15m/classification.h5`
- Training Code: `src/stage2_trainer.py`
- Inference Code: `src/stage2_inference.py`
- Data: `data/stage2/`

---

**Created:** Jan 5, 2026
**Last Updated:** Jan 5, 2026
