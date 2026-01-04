import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict

class VolatilityModel:
    """
    Model to predict volatility regimes (large move vs ranging).
    """
    
    def __init__(self, atr_multiplier: float = 1.5, atr_window: int = 14):
        """
        Initialize volatility model.
        
        Args:
            atr_multiplier: Multiplier for ATR to define large move threshold
            atr_window: Window for ATR calculation
        """
        self.atr_multiplier = atr_multiplier
        self.atr_window = atr_window
        self.model = None
        self.scaler = StandardScaler()
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range.
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = tr.rolling(window=self.atr_window).mean()
        
        return atr
    
    def create_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create binary labels for volatility regime.
        
        0: Ranging market (low volatility)
        1: Large move (high volatility)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Binary labels
        """
        atr = self._calculate_atr(df)
        atr_ma = atr.rolling(window=20).mean()
        
        # Large move if current ATR > mean ATR * multiplier
        labels = (atr > atr_ma * self.atr_multiplier).astype(int).fillna(0)
        
        return labels.values
    
    def create_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create features for volatility prediction.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Feature array
        """
        features = []
        
        # ATR and related
        atr = self._calculate_atr(df)
        features.append(atr.fillna(0).values)
        
        atr_ma = atr.rolling(window=20).mean()
        features.append(atr_ma.fillna(0).values)
        
        features.append((atr / atr_ma).fillna(1).values)
        
        # Range
        range_pct = ((df['high'] - df['low']) / df['close']).fillna(0).values
        features.append(range_pct)
        
        # Volatility (std of returns)
        for period in [5, 10, 20]:
            volatility = df['close'].pct_change().rolling(window=period).std().fillna(0).values
            features.append(volatility)
        
        # Volume related
        if 'volume' in df.columns:
            vol_ma = df['volume'].rolling(window=20).mean()
            features.append((df['volume'] / vol_ma).fillna(1).values)
            
            vol_change = df['volume'].pct_change().fillna(0).values
            features.append(vol_change)
        
        # Close position in high-low range
        close_pos = ((df['close'] - df['low']) / (df['high'] - df['low'])).fillna(0.5).values
        features.append(close_pos)
        
        # Body size
        body_size = (abs(df['close'] - df['open']) / (df['high'] - df['low'])).fillna(0).values
        features.append(body_size)
        
        # Price changes
        for period in [1, 5, 10]:
            price_change = df['close'].pct_change(period).fillna(0).values
            features.append(price_change)
        
        features_array = np.column_stack(features)
        return features_array
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Train volatility model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training metrics
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train LightGBM
        self.model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1
        )
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Evaluate
        y_pred_val = self.model.predict(X_val_scaled)
        
        metrics = {
            'val_accuracy': accuracy_score(y_val, y_pred_val),
            'val_precision': precision_score(y_val, y_pred_val, zero_division=0),
            'val_recall': recall_score(y_val, y_pred_val, zero_division=0),
            'val_f1': f1_score(y_val, y_pred_val, zero_division=0)
        }
        
        return metrics
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict volatility regime.
        
        Args:
            X_test: Test features
            
        Returns:
            Predictions and probabilities
        """
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        probabilities = self.model.predict_proba(X_test_scaled)
        
        return predictions, probabilities
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        predictions, _ = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1': f1_score(y_test, predictions, zero_division=0)
        }
        
        return metrics
