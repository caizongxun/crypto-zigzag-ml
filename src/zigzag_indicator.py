import numpy as np
import pandas as pd
from typing import Tuple, List

class ZigZagIndicator:
    """
    ZigZag indicator implementation based on MT4 algorithm.
    Identifies pivot points and labels them as HH, HL, LL, LH.
    """
    
    def __init__(self, depth: int = 12, deviation: int = 5, backstep: int = 2):
        """
        Initialize ZigZag indicator.
        
        Args:
            depth: Number of bars to look back for pivot identification
            deviation: Minimum percentage change to consider as pivot
            backstep: Minimum bars between pivots
        """
        self.depth = depth
        self.deviation = deviation
        self.backstep = backstep
    
    def detect_pivots(self, high: np.ndarray, low: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect pivot points (local highs and lows).
        
        Args:
            high: Array of high prices
            low: Array of low prices
            
        Returns:
            pivot_highs: Array of high pivot indices (0 if not pivot)
            pivot_lows: Array of low pivot indices (0 if not pivot)
        """
        n = len(high)
        pivot_highs = np.zeros(n, dtype=np.float64)
        pivot_lows = np.zeros(n, dtype=np.float64)
        
        for i in range(self.depth, n - self.depth):
            # Check for local high
            if high[i] >= np.max(high[i - self.depth:i + self.depth]):
                pivot_highs[i] = high[i]
            
            # Check for local low
            if low[i] <= np.min(low[i - self.depth:i + self.depth]):
                pivot_lows[i] = low[i]
        
        return pivot_highs, pivot_lows
    
    def identify_direction_changes(self, high: np.ndarray, low: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Identify zigzag direction and pivots based on deviation threshold.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            
        Returns:
            direction: Array indicating direction (1 for up, -1 for down)
            pivots: List of pivot information {index, price, type}
        """
        n = len(high)
        direction = np.zeros(n, dtype=np.int8)
        pivots = []
        
        last_pivot_price = low[0]
        last_pivot_idx = 0
        current_direction = 1
        
        for i in range(1, n):
            if current_direction == 1:
                # Looking for higher high
                if high[i] > last_pivot_price * (1 + self.deviation / 100):
                    # Continue uptrend
                    direction[i] = 1
                elif low[i] < last_pivot_price * (1 - self.deviation / 100):
                    # Reversal to downtrend
                    if i - last_pivot_idx >= self.backstep:
                        pivots.append({
                            'index': i,
                            'price': high[last_pivot_idx],
                            'type': 'high'
                        })
                        last_pivot_price = high[last_pivot_idx]
                        last_pivot_idx = i
                        current_direction = -1
                        direction[i] = -1
                else:
                    direction[i] = current_direction
            else:
                # Looking for lower low
                if low[i] < last_pivot_price * (1 - self.deviation / 100):
                    # Continue downtrend
                    direction[i] = -1
                elif high[i] > last_pivot_price * (1 + self.deviation / 100):
                    # Reversal to uptrend
                    if i - last_pivot_idx >= self.backstep:
                        pivots.append({
                            'index': i,
                            'price': low[last_pivot_idx],
                            'type': 'low'
                        })
                        last_pivot_price = low[last_pivot_idx]
                        last_pivot_idx = i
                        current_direction = 1
                        direction[i] = 1
                else:
                    direction[i] = current_direction
        
        return direction, pivots
    
    def label_kbars(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label K-bars with zigzag signal patterns.
        
        Labels:
            0: No signal
            1: HH (Higher High) - Short signal
            2: LH (Lower High) - Short signal
            3: HL (Higher Low) - Long signal
            4: LL (Lower Low) - Long signal
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            df: DataFrame with 'zigzag_label' column added
        """
        df = df.copy()
        high = df['high'].values
        low = df['low'].values
        n = len(df)
        
        labels = np.zeros(n, dtype=np.int8)
        last_pivot_type = None
        last_pivot_price = None
        
        # Find initial pivot
        for i in range(self.depth, min(self.depth * 3, n)):
            if high[i] >= np.max(high[max(0, i - self.depth):i + 1]):
                last_pivot_type = 'high'
                last_pivot_price = high[i]
                break
        
        if last_pivot_type is None:
            for i in range(self.depth, min(self.depth * 3, n)):
                if low[i] <= np.min(low[max(0, i - self.depth):i + 1]):
                    last_pivot_type = 'low'
                    last_pivot_price = low[i]
                    break
        
        # Process remaining bars
        for i in range(self.depth, n):
            if last_pivot_type == 'high':
                current_high = high[i]
                # Check for new high
                if current_high > last_pivot_price * (1 + self.deviation / 100):
                    labels[i] = 1  # HH (Higher High)
                    last_pivot_price = current_high
                # Check for reversal to low
                elif low[i] < last_pivot_price * (1 - self.deviation / 100):
                    labels[i] = 3  # HL (Higher Low)
                    last_pivot_type = 'low'
                    last_pivot_price = low[i]
            else:  # last_pivot_type == 'low'
                current_low = low[i]
                # Check for new low
                if current_low < last_pivot_price * (1 - self.deviation / 100):
                    labels[i] = 4  # LL (Lower Low)
                    last_pivot_price = current_low
                # Check for reversal to high
                elif high[i] > last_pivot_price * (1 + self.deviation / 100):
                    labels[i] = 2  # LH (Lower High)
                    last_pivot_type = 'high'
                    last_pivot_price = high[i]
        
        df['zigzag_label'] = labels
        return df
    
    def get_label_name(self, label: int) -> str:
        """
        Convert label integer to string representation.
        
        Args:
            label: Integer label (0-4)
            
        Returns:
            String representation of the label
        """
        label_map = {
            0: 'NO_SIGNAL',
            1: 'HH',
            2: 'LH',
            3: 'HL',
            4: 'LL'
        }
        return label_map.get(label, 'UNKNOWN')
    
    def get_signal_type(self, label: int) -> str:
        """
        Get trading signal type from label.
        
        Args:
            label: Integer label (0-4)
            
        Returns:
            Signal type: 'SHORT', 'LONG', or 'NONE'
        """
        if label in [1, 2]:  # HH, LH
            return 'SHORT'
        elif label in [3, 4]:  # HL, LL
            return 'LONG'
        else:
            return 'NONE'
