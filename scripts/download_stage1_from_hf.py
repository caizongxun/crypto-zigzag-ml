#!/usr/bin/env python3
"""
Download all Stage 1 classification models from HuggingFace

Usage:
    python scripts/download_stage1_from_hf.py                  # Download all
    python scripts/download_stage1_from_hf.py --symbols BTC    # Download only BTC
    python scripts/download_stage1_from_hf.py --timeframe 15m  # Download only 15m
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple
import json
from huggingface_hub import hf_hub_download
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

HF_DATASET_ID = 'zongowo111/v2-crypto-ohlcv-data'
HF_MODEL_BASE = 'v1_model'

ALL_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT',
    'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT', 'POLYUSDT', 'LINKUSDT', 'LTCUSDT',
    'BCHUSDT', 'UNIUSDT', 'ATOMUSDT', 'ALGOUSDT', 'FILUSDT', 'OPUSDT',
    'ARBUSDT', 'MATICUSDT', 'AAVEUSDT', 'NEARUSDT'
]

TIMEFRAMES = ['15m', '1h']

MODEL_FILES = ['classification.h5', 'params.json']


class Stage1ModelDownloader:
    """
    Download Stage 1 models from HuggingFace
    """
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.download_stats = {
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'total_size_mb': 0
        }
    
    def download_model(
        self,
        symbol: str,
        timeframe: str,
        force: bool = False
    ) -> bool:
        """
        Download Stage 1 model for a specific symbol and timeframe
        
        Parameters:
        -----------
        symbol : str
            Symbol name (e.g., 'BTCUSDT')
        timeframe : str
            Timeframe (e.g., '15m' or '1h')
        force : bool
            Force download even if files exist
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        
        local_dir = self.model_dir / 'stage1' / f'{symbol.lower()}_{timeframe}'
        hf_path = f'{HF_MODEL_BASE}/{symbol}/{timeframe}'
        
        # Check if already downloaded
        if not force:
            existing_files = list(local_dir.glob('*.h5')) + list(local_dir.glob('*.json'))
            if len(existing_files) >= 2:
                logger.info(f'Skipping {symbol} {timeframe} (already exists)')
                self.download_stats['skipped'] += 1
                return True
        
        local_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f'Downloading {symbol} {timeframe}...')
        
        for filename in MODEL_FILES:
            try:
                logger.debug(f'  Downloading {filename}...')
                file_path = hf_hub_download(
                    repo_id=HF_DATASET_ID,
                    filename=f'{hf_path}/{filename}',
                    repo_type='dataset',
                    cache_dir=str(local_dir)
                )
                
                # Get file size
                size_mb = Path(file_path).stat().st_size / 1024 / 1024
                self.download_stats['total_size_mb'] += size_mb
                
                logger.debug(f'    {filename} ({size_mb:.2f} MB)')
                
            except Exception as e:
                logger.error(f'  Error downloading {filename}: {str(e)}')
                self.download_stats['failed'] += 1
                return False
        
        logger.info(f'✓ Successfully downloaded {symbol} {timeframe}')
        self.download_stats['success'] += 1
        return True
    
    def download_all(
        self,
        symbols: List[str] = None,
        timeframes: List[str] = None,
        force: bool = False
    ):
        """
        Download all Stage 1 models
        
        Parameters:
        -----------
        symbols : List[str]
            Symbols to download (default: all)
        timeframes : List[str]
            Timeframes to download (default: all)
        force : bool
            Force download even if files exist
        """
        
        if symbols is None:
            symbols = ALL_SYMBOLS
        if timeframes is None:
            timeframes = TIMEFRAMES
        
        logger.info(f'Starting download of {len(symbols)} symbols × {len(timeframes)} timeframes')
        logger.info(f'Dataset: {HF_DATASET_ID}')
        logger.info(f'Target directory: {self.model_dir}')
        
        total = len(symbols) * len(timeframes)
        current = 0
        
        for symbol in symbols:
            for timeframe in timeframes:
                current += 1
                logger.info(f'[{current}/{total}] Downloading {symbol} {timeframe}...')
                
                try:
                    self.download_model(symbol, timeframe, force)
                except Exception as e:
                    logger.error(f'Unexpected error: {str(e)}')
                    self.download_stats['failed'] += 1
        
        self._print_summary()
    
    def _print_summary(self):
        """
        Print download summary
        """
        
        print(f'\n' + '='*80)
        print(f'DOWNLOAD SUMMARY')
        print(f'='*80)
        print(f'Success: {self.download_stats["success"]}')
        print(f'Failed: {self.download_stats["failed"]}')
        print(f'Skipped: {self.download_stats["skipped"]}')
        print(f'Total Size: {self.download_stats["total_size_mb"]:.2f} MB')
        print(f'='*80)


def main():
    parser = argparse.ArgumentParser(
        description='Download Stage 1 models from HuggingFace'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated symbols (e.g., BTCUSDT,ETHUSDT) or single symbol (e.g., BTC)',
        default=None
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        choices=['15m', '1h'],
        help='Timeframe to download',
        default=None
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force download even if files exist'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Local directory to save models'
    )
    
    args = parser.parse_args()
    
    downloader = Stage1ModelDownloader(model_dir=args.model_dir)
    
    # Parse symbols
    symbols = ALL_SYMBOLS
    if args.symbols:
        if ',' in args.symbols:
            symbols = args.symbols.split(',')
        else:
            # Match by prefix (e.g., 'BTC' -> 'BTCUSDT')
            symbols = [s for s in ALL_SYMBOLS if s.startswith(args.symbols.upper())]
    
    # Parse timeframes
    timeframes = TIMEFRAMES
    if args.timeframe:
        timeframes = [args.timeframe]
    
    logger.info(f'Symbols to download: {symbols}')
    logger.info(f'Timeframes to download: {timeframes}')
    
    # Download
    downloader.download_all(symbols, timeframes, force=args.force)


if __name__ == '__main__':
    main()
