import numpy as np
import pandas as pd
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering for crypto trading signals.
    Generates 55+ technical indicators and price action features.
    """
    
    def __init__(self, lookback_periods: List[int] = [5, 10, 20, 50, 200]):
        """
        Initialize feature engineer.
        
        Args:
            lookback_periods: Periods for moving averages and other indicators
        """
        self.lookback_periods = lookback_periods
        self.feature_count = 0
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all 55+ features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features added
        """
        df = df.copy()
        
        # Price action features (10-15 features)
        df = self._add_price_action_features(df)
        
        # Moving averages and trends (15-20 features)
        df = self._add_moving_average_features(df)
        
        # Volatility features (5-10 features)
        df = self._add_volatility_features(df)
        
        # Momentum features (10-15 features)
        df = self._add_momentum_features(df)
        
        # Volume features (5-10 features)
        df = self._add_volume_features(df)
        
        # Support/Resistance features (3-5 features)
        df = self._add_support_resistance_features(df)
        
        # Structure features (3-5 features)
        df = self._add_structure_features(df)
        
        return df
    
    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price action features.
        """
        # High-low range
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['open_close_range'] = (df['close'] - df['open']) / df['open']
        
        # True range
        df['true_range'] = self._calculate_true_range(df)
        
        # Price changes
        for period in [1, 2, 5, 10]:
            df[f'price_change_{period}'] = df['close'].pct_change(period)
            df[f'high_change_{period}'] = df['high'].pct_change(period)
            df[f'low_change_{period}'] = df['low'].pct_change(period)
        
        # Close position relative to high-low
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        # Body size
        df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        
        return df
    
    def _add_moving_average_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add moving average and trend features.
        """
        # Filter lookback periods based on data length
        max_period = len(df) // 3
        valid_periods = [p for p in self.lookback_periods if p < max_period]
        
        # If no valid periods, use minimal ones
        if not valid_periods:
            valid_periods = [5, 10]
        
        for period in valid_periods:
            # Simple moving averages
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'sma_ratio_{period}'] = df['close'] / (df[f'sma_{period}'] + 1e-8)
            
            # Exponential moving averages
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'ema_ratio_{period}'] = df['close'] / (df[f'ema_{period}'] + 1e-8)
            
            # Distance to MAs
            df[f'distance_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / (df[f'sma_{period}'] + 1e-8)
        
        # Trend strength (only if we have enough periods)
        if len(valid_periods) >= 2 and valid_periods[-1] >= 20:
            idx_10 = valid_periods[0] if valid_periods[0] == 10 else (1 if len(valid_periods) > 1 else 0)
            idx_20 = next((i for i, p in enumerate(valid_periods) if p == 20), None)
            idx_50 = next((i for i, p in enumerate(valid_periods) if p >= 50), None)
            
            if idx_10 is not None and idx_20 is not None:
                p1, p2 = valid_periods[idx_10], valid_periods[idx_20]
                df['trend_strength_10'] = (df[f'sma_{p1}'] - df[f'sma_{p2}']) / (df['close'] + 1e-8)
            
            if idx_20 is not None and idx_50 is not None:
                p1, p2 = valid_periods[idx_20], valid_periods[idx_50]
                df['trend_strength_20'] = (df[f'sma_{p1}'] - df[f'sma_{p2}']) / (df['close'] + 1e-8)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility features.
        """
        # ATR (Average True Range)
        df['atr_14'] = self._calculate_atr(df, period=14)
        df['atr_ratio'] = df['atr_14'] / df['close']
        
        # Historical volatility
        for period in [5, 10, 20]:
            if period < len(df) // 3:
                df[f'volatility_{period}'] = df['close'].pct_change().rolling(window=period).std()
        
        # Bollinger Bands
        df = self._add_bollinger_bands(df)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum features.
        """
        # RSI (Relative Strength Index)
        df['rsi_14'] = self._calculate_rsi(df, period=14)
        df['rsi_7'] = self._calculate_rsi(df, period=7)
        
        # MACD
        df = self._add_macd(df)
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / (df['close'].shift(period) + 1e-8)
        
        # Stochastic Oscillator
        df = self._add_stochastic(df)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume features.
        """
        if 'volume' in df.columns:
            # Volume ratios
            df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma_10'] + 1e-8)
            
            # Volume trend
            for period in [5, 10, 20]:
                df[f'volume_change_{period}'] = df['volume'].pct_change(period)
            
            # On-Balance Volume
            df['obv'] = ((np.sign(df['close'].diff())) * df['volume']).fillna(0).cumsum()
            df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        
        return df
    
    def _add_support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add support/resistance level features.
        """
        # Highest high and lowest low
        for period in [10, 20, 50]:
            if period < len(df) // 3:
                df[f'highest_high_{period}'] = df['high'].rolling(window=period).max()
                df[f'lowest_low_{period}'] = df['low'].rolling(window=period).min()
                
                # Distance to support/resistance
                df[f'distance_resistance_{period}'] = (df[f'highest_high_{period}'] - df['close']) / df['close']
                df[f'distance_support_{period}'] = (df['close'] - df[f'lowest_low_{period}']) / df['close']
        
        return df
    
    def _add_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market structure features.
        """
        # Consecutive up/down bars
        up_down = np.where(df['close'] > df['open'], 1, -1)
        df['consecutive_bars'] = up_down * (up_down.groupby((up_down != up_down.shift()).cumsum()).cumcount() + 1)
        
        # Swing highs/lows
        df['swing_high_5'] = df['high'].rolling(window=5, center=True).max() == df['high']
        df['swing_low_5'] = df['low'].rolling(window=5, center=True).min() == df['low']
        
        return df
    
    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate true range.
        """
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        """
        tr = self._calculate_true_range(df)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        """
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add MACD features.
        """
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Bands features.
        """
        if period < len(df) // 3:
            df[f'bb_mid_{period}'] = df['close'].rolling(window=period).mean()
            bb_std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = df[f'bb_mid_{period}'] + (std_dev * bb_std)
            df[f'bb_lower_{period}'] = df[f'bb_mid_{period}'] - (std_dev * bb_std)
            
            # BB percentage
            df[f'bb_percent_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / \
                                          (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-8)
        
        return df
    
    def _add_stochastic(self, df: pd.DataFrame, period: int = 14, smooth_k: int = 3) -> pd.DataFrame:
        """
        Add Stochastic Oscillator features.
        """
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        
        df['stoch_raw'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-8)
        df['stoch_k'] = df['stoch_raw'].rolling(window=smooth_k).mean()
        df['stoch_d'] = df['stoch_k'].rolling(window=smooth_k).mean()
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of all feature columns (excluding OHLCV and labels).
        
        Args:
            df: DataFrame with features
            
        Returns:
            List of feature column names
        """
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'date', 'timestamp', 'zigzag_label']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
