# Crypto ZigZag ML Implementation Guide

Step-by-step guide to implement and train the complete system.

## Phase 1: Environment Setup

```bash
# Clone repository
git clone https://github.com/caizongxun/crypto-zigzag-ml
cd crypto-zigzag-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## Phase 2: Test Basic Components

```bash
# Run tests
python tests/test_zigzag.py

# Output should show:
# - Test 1: ZigZag basic functionality - Success!
# - Test 2: Feature engineering - Success!
# - All tests passed!
```

## Phase 3: Fetch and Explore Data

```bash
# Fetch BTC_15m test data
python data/fetch_data.py

# This will download data from Hugging Face and show:
# - BTC_15m shape
# - Columns and date range
# - First 5 rows
```

## Phase 4: Jupyter Notebooks (Sequential)

### Notebook 1: Data Exploration
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```
Objectives:
- Fetch BTC_15m data
- Visualize price and volume
- Check data quality

### Notebook 2: Feature Engineering
```bash
jupyter notebook notebooks/02_feature_engineering.ipynb
```
Objectives:
- Apply ZigZag indicator
- Generate 55+ features
- Validate label distribution
- Check for missing values

### Notebook 3: Model Training
```bash
jupyter notebook notebooks/03_model_training.ipynb
```
Objectives:
- Train LSTM + XGBoost model
- Train volatility prediction model
- Evaluate performance
- Check if accuracy meets 80-90% target

## Phase 5: Key Parameters to Adjust

### config.yaml - ZigZag Parameters
```yaml
zigzag:
  depth: 12          # Increase for less frequent signals
  deviation: 5       # Percentage change threshold
  backstep: 2        # Minimum bars between pivots
```

### config.yaml - Model Parameters
```yaml
lstm_config:
  timesteps: 60      # Historical bars to use
  lstm_units: [128, 64]  # Layer sizes
  dropout: 0.3       # Increase if overfitting
  epochs: 100
  early_stopping_patience: 10

xgboost_config:
  n_estimators: 150
  max_depth: 7       # Decrease to prevent overfitting
  learning_rate: 0.05
  subsample: 0.8     # Increase for less overfitting
```

## Phase 6: What to Monitor

### Performance Targets
- Model Accuracy: 80-90%
- Precision: >80%
- Recall: >75%
- Sharpe Ratio: >1.0
- Max Drawdown: <20%

### Anti-Overfitting Checks
1. Compare train vs validation accuracy gap
   - If gap > 10%, increase dropout or reduce depth
2. Check test set performance
   - If test < val, model is overfitting
3. Monitor class distribution
   - Imbalanced labels need weighted loss

## Phase 7: Multi-Symbol Expansion

Once BTC_15m is working, scale to other symbols:

```python
from data.fetch_data import CryptoDataFetcher

fetcher = CryptoDataFetcher()
all_data = fetcher.fetch_all_symbols(timeframes=['15m', '1h'])

# all_data now contains data for all 22 symbols
```

## Phase 8: Troubleshooting

### Issue: Model accuracy stuck at ~50%
**Solution:**
- Check label distribution (should be reasonably balanced)
- Verify ZigZag parameters produce meaningful signals
- Increase model complexity (more LSTM units)

### Issue: Model overfitting (train: 95%, val: 60%)
**Solution:**
- Increase dropout (0.3 -> 0.4-0.5)
- Reduce model depth
- Reduce learning rate
- Add more regularization

### Issue: Features contain NaN values
**Solution:**
- Forward fill: `df.fillna(method='ffill')`
- Or drop rows: `df.dropna()`
- Check lookback periods aren't too large

### Issue: Data fetch fails
**Solution:**
- Verify internet connection
- Check HuggingFace dataset URL
- Install huggingface_hub: `pip install huggingface-hub`

## Next Steps After Training

1. Save trained models
2. Create Flask API for real-time predictions
3. Backtest on different symbols/timeframes
4. Deploy to production with risk management
5. Monitor model performance in live trading

## Expected Timeline

- Phase 1-2: 10 minutes
- Phase 3: 15 minutes (first download)
- Phase 4: 1-2 hours (exploration)
- Phase 5-6: 2-4 hours (tuning)
- Phase 7: 2-3 hours (multi-symbol)

Total: 6-10 hours for complete setup and initial training
