# Crypto ZigZag ML Trading System

A comprehensive machine learning system for cryptocurrency trading based on ZigZag pattern recognition (HH/HL/LL/LH signals) with volatility prediction across 22 crypto symbols.

## Project Overview

This project implements a complete ML pipeline for:
1. ZigZag pattern detection and K-bar labeling
2. Feature engineering (50-60 technical indicators and price action features)
3. Hybrid LSTM + XGBoost model for signal classification (80-90% target accuracy)
4. Volatility regime prediction (large move vs ranging markets)
5. Backtesting and performance analysis

## Target Specifications

- **Crypto symbols**: 22 cryptocurrencies (starting with BTC_15m)
- **Timeframe**: 15m and 1h OHLCV data
- **Data source**: Hugging Face dataset (https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data)
- **Model accuracy target**: 80-90%
- **Anti-overfitting measures**: Time-series split, regularization, early stopping

## Project Structure

```
crypto-zigzag-ml/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── fetch_data.py
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── zigzag_indicator.py
│   ├── features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_xgboost.py
│   │   └── volatility_model.py
│   ├── utils.py
│   └── config.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── backtest/
│   └── backtest.py
└── tests/
    └── test_zigzag.py
```

## Installation

```bash
git clone https://github.com/caizongxun/crypto-zigzag-ml
cd crypto-zigzag-ml
pip install -r requirements.txt
```

## Usage

See step-by-step instructions in notebooks/

## Training Pipeline

1. Data fetching and exploration
2. ZigZag indicator implementation and labeling
3. Feature engineering (50-60 features)
4. Model training (LSTM + XGBoost)
5. Volatility prediction model
6. Backtesting and evaluation

## References

- ZigZag indicator based on MT4 implementation
- LSTM + XGBoost hybrid architecture for time-series classification
- Time-series cross-validation to prevent data leakage
