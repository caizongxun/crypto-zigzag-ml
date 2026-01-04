import os
import pandas as pd
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import config

class CryptoDataFetcher:
    """
    Fetch cryptocurrency OHLCV data from Hugging Face.
    """
    
    def __init__(self, dataset_name: str = 'zongowo111/v2-crypto-ohlcv-data'):
        """
        Initialize data fetcher.
        
        Args:
            dataset_name: HuggingFace dataset name
        """
        self.dataset_name = dataset_name
        self.cache_dir = project_root / 'data' / 'raw'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_symbol_timeframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Fetch data for specific symbol and timeframe.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '15m', '1h')
            
        Returns:
            DataFrame with OHLCV data
        """
        # File name in HF format: BTC_15m.parquet
        symbol_base = symbol.replace('USDT', '')
        filename = f'{symbol_base}_{timeframe}.parquet'
        
        try:
            filepath = hf_hub_download(
                repo_id=self.dataset_name,
                filename=f'klines/{symbol}/{filename}',
                repo_type='dataset',
                cache_dir=str(self.cache_dir)
            )
            
            df = pd.read_parquet(filepath)
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Filter and clean data
            df = self._clean_data(df)
            
            return df
        
        except Exception as e:
            print(f'Error fetching {symbol} {timeframe}: {str(e)}')
            return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names.
        """
        df = df.copy()
        column_mapping = {
            'open_time': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'open_time': 'timestamp'
        }
        
        df.columns = df.columns.str.lower()
        df = df.rename(columns=column_mapping)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate OHLCV data.
        """
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove rows with NaN in critical columns
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # Validate OHLC relationships
        df = df[(df['high'] >= df['low']) & 
               (df['high'] >= df['close']) & 
               (df['high'] >= df['open']) &
               (df['low'] <= df['close']) & 
               (df['low'] <= df['open'])]
        
        return df.reset_index(drop=True)
    
    def fetch_all_symbols(self, timeframes: list = None) -> dict:
        """
        Fetch data for all configured symbols and timeframes.
        
        Args:
            timeframes: List of timeframes to fetch
            
        Returns:
            Dictionary {symbol: {timeframe: DataFrame}}
        """
        if timeframes is None:
            timeframes = config['data']['timeframes']
        
        symbols = config['crypto_symbols']
        all_data = {}
        
        for symbol in symbols:
            all_data[symbol] = {}
            for timeframe in timeframes:
                print(f'Fetching {symbol} {timeframe}...')
                df = self.fetch_symbol_timeframe(symbol, timeframe)
                if df is not None:
                    all_data[symbol][timeframe] = df
                    print(f'  Success: {len(df)} rows')
                else:
                    print(f'  Failed')
        
        return all_data

if __name__ == '__main__':
    fetcher = CryptoDataFetcher()
    
    print('Fetching BTC_15m as test...')
    btc_15m = fetcher.fetch_symbol_timeframe('BTCUSDT', '15m')
    
    if btc_15m is not None:
        print(f'\nBTC_15m shape: {btc_15m.shape}')
        print(f'Columns: {btc_15m.columns.tolist()}')
        print(f'Date range: {btc_15m["timestamp"].min()} to {btc_15m["timestamp"].max()}')
        print(f'\nFirst 5 rows:')
        print(btc_15m.head())
    else:
        print('Failed to fetch data')
