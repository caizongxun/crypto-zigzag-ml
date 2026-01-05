import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Tuple
import logging
from datetime import datetime

from src.stage2_trainer import Stage2Trainer
from data.fetch_data import CryptoDataFetcher
from src.zigzag_indicator import ZigZagIndicator
from src.features import FeatureEngineer
from src.utils import time_series_split, normalize_data
from tensorflow import keras

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Stage2BatchTrainer:
    """
    Batch trainer for Stage 2 models across multiple symbols and timeframes
    """
    
    def __init__(self, model_dir='models', data_dir='data'):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.results = {}
        
    def prepare_all_data(self, symbols_timeframes: List[Tuple[str, str]],
                        depth=12, deviation=5, backstep=2):
        """
        Prepare Stage 2 data for all symbols
        
        Parameters:
        -----------
        symbols_timeframes : List[Tuple[str, str]]
            List of (symbol, timeframe) pairs
            e.g., [('BTCUSDT', '15m'), ('ETHUSDT', '1h')]
        """
        
        logger.info(f"Preparing data for {len(symbols_timeframes)} symbols...")
        
        for symbol, timeframe in symbols_timeframes:
            try:
                logger.info(f"Processing {symbol} {timeframe}...")
                self._prepare_data_single(symbol, timeframe, depth, deviation, backstep)
            except Exception as e:
                logger.error(f"Error processing {symbol} {timeframe}: {str(e)}")
                continue
        
        logger.info("Data preparation complete!")
    
    def _prepare_data_single(self, symbol: str, timeframe: str,
                             depth: int, deviation: int, backstep: int):
        """
        Prepare Stage 2 data for single symbol-timeframe
        """
        
        fetcher = CryptoDataFetcher()
        df = fetcher.fetch_symbol_timeframe(symbol, timeframe)
        
        if df is None or len(df) == 0:
            logger.warning(f"No data found for {symbol} {timeframe}")
            return
        
        zigzag = ZigZagIndicator(depth=depth, deviation=deviation, backstep=backstep)
        df = zigzag.label_kbars(df)
        
        fe = FeatureEngineer(lookback_periods=[5, 10, 20, 50, 200])
        df = fe.calculate_all_features(df)
        
        feature_cols = fe.get_feature_columns(df)
        if 'symbol' in feature_cols:
            feature_cols.remove('symbol')
        
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0)
        
        train_df, val_df, test_df = time_series_split(df, train_ratio=0.7, validation_ratio=0.15)
        
        stage1_model = keras.models.load_model(
            str(self.model_dir / f'{symbol.lower()}_{timeframe}' / 'classification.h5')
        )
        
        X_train = train_df[feature_cols].values
        y_train = train_df['zigzag_label'].values
        stage1_probs_train = stage1_model.predict(X_train, verbose=0)
        stage1_preds_train = (stage1_probs_train[:, 1] > 0.5).astype(int)
        
        signal_mask = stage1_preds_train == 1
        X_stage2_train = X_train[signal_mask]
        y_stage2_train = y_train[signal_mask]
        valid_mask = y_stage2_train > 0
        X_stage2_train = X_stage2_train[valid_mask]
        y_stage2_train = y_stage2_train[valid_mask]
        
        X_val = val_df[feature_cols].values
        y_val = val_df['zigzag_label'].values
        stage1_probs_val = stage1_model.predict(X_val, verbose=0)
        stage1_preds_val = (stage1_probs_val[:, 1] > 0.5).astype(int)
        
        signal_mask_val = stage1_preds_val == 1
        X_stage2_val = X_val[signal_mask_val]
        y_stage2_val = y_val[signal_mask_val]
        valid_mask_val = y_stage2_val > 0
        X_stage2_val = X_stage2_val[valid_mask_val]
        y_stage2_val = y_stage2_val[valid_mask_val]
        
        X_test = test_df[feature_cols].values
        y_test = test_df['zigzag_label'].values
        stage1_probs_test = stage1_model.predict(X_test, verbose=0)
        stage1_preds_test = (stage1_probs_test[:, 1] > 0.5).astype(int)
        
        signal_mask_test = stage1_preds_test == 1
        X_stage2_test = X_test[signal_mask_test]
        y_stage2_test = y_test[signal_mask_test]
        valid_mask_test = y_stage2_test > 0
        X_stage2_test = X_stage2_test[valid_mask_test]
        y_stage2_test = y_stage2_test[valid_mask_test]
        
        data_dir = self.data_dir / 'stage2' / f'{symbol.lower()}_{timeframe}'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        with open(data_dir / 'X_stage2_train.pkl', 'wb') as f:
            pickle.dump(X_stage2_train, f)
        with open(data_dir / 'y_stage2_train.pkl', 'wb') as f:
            pickle.dump(y_stage2_train, f)
        with open(data_dir / 'X_stage2_val.pkl', 'wb') as f:
            pickle.dump(X_stage2_val, f)
        with open(data_dir / 'y_stage2_val.pkl', 'wb') as f:
            pickle.dump(y_stage2_val, f)
        with open(data_dir / 'X_stage2_test.pkl', 'wb') as f:
            pickle.dump(X_stage2_test, f)
        with open(data_dir / 'y_stage2_test.pkl', 'wb') as f:
            pickle.dump(y_stage2_test, f)
        
        logger.info(f"  Train: {len(X_stage2_train)}, Val: {len(X_stage2_val)}, Test: {len(X_stage2_test)}")
    
    def train_all(self, symbols_timeframes: List[Tuple[str, str]],
                  cv_folds: int = 5):
        """
        Train Stage 2 models for all symbols
        
        Parameters:
        -----------
        symbols_timeframes : List[Tuple[str, str]]
        cv_folds : int
        """
        
        logger.info(f"Training Stage 2 models for {len(symbols_timeframes)} symbols...")
        
        for symbol, timeframe in symbols_timeframes:
            try:
                logger.info(f"\nTraining {symbol} {timeframe}...")
                self._train_single(symbol, timeframe, cv_folds)
            except Exception as e:
                logger.error(f"Error training {symbol} {timeframe}: {str(e)}")
                self.results[f"{symbol}_{timeframe}"] = {
                    'status': 'failed',
                    'error': str(e)
                }
                continue
        
        self._print_summary()
    
    def _train_single(self, symbol: str, timeframe: str, cv_folds: int):
        """
        Train Stage 2 model for single symbol-timeframe
        """
        
        data_dir = self.data_dir / 'stage2' / f'{symbol.lower()}_{timeframe}'
        
        with open(data_dir / 'X_stage2_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        with open(data_dir / 'y_stage2_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
        with open(data_dir / 'X_stage2_val.pkl', 'rb') as f:
            X_val = pickle.load(f)
        with open(data_dir / 'y_stage2_val.pkl', 'rb') as f:
            y_val = pickle.load(f)
        with open(data_dir / 'X_stage2_test.pkl', 'rb') as f:
            X_test = pickle.load(f)
        with open(data_dir / 'y_stage2_test.pkl', 'rb') as f:
            y_test = pickle.load(f)
        
        trainer = Stage2Trainer(model_dir=str(self.model_dir / 'stage2'))
        
        train_results = trainer.train(
            X_train, y_train,
            X_val, y_val,
            normalize=True,
            cv_folds=cv_folds,
            save_model=True
        )
        
        test_metrics = trainer.evaluate(X_test, y_test)
        
        cv_results = trainer.cross_validate(
            np.vstack([X_train, X_val]),
            np.hstack([y_train, y_val]),
            cv=cv_folds
        )
        
        self.results[f"{symbol}_{timeframe}"] = {
            'status': 'success',
            'train_accuracy': train_results['train_accuracy'],
            'val_accuracy': train_results['val_accuracy'],
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1_score'],
            'cv_mean_accuracy': cv_results['mean_accuracy'],
            'cv_std_accuracy': cv_results['std_accuracy'],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Test F1-Score: {test_metrics['f1_score']:.4f}")
        logger.info(f"  CV Mean: {cv_results['mean_accuracy']:.4f} +/- {cv_results['std_accuracy']:.4f}")
    
    def _print_summary(self):
        """
        Print training summary
        """
        
        print("\n" + "="*80)
        print("STAGE 2 BATCH TRAINING SUMMARY")
        print("="*80)
        print(f"\n{'Symbol':<12} {'Status':<10} {'Test Acc':<10} {'Test F1':<10} {'CV Mean':<10}")
        print("-"*80)
        
        successful = 0
        failed = 0
        
        for key, result in sorted(self.results.items()):
            status = result['status']
            
            if status == 'success':
                successful += 1
                test_acc = result['test_accuracy']
                test_f1 = result['test_f1']
                cv_mean = result['cv_mean_accuracy']
                print(f"{key:<12} {status:<10} {test_acc:<10.4f} {test_f1:<10.4f} {cv_mean:<10.4f}")
            else:
                failed += 1
                print(f"{key:<12} {status:<10} {result.get('error', 'Unknown error'):<50}")
        
        print("-"*80)
        print(f"\nTotal: {successful} successful, {failed} failed")
        
        if successful > 0:
            avg_test_acc = np.mean([r['test_accuracy'] for r in self.results.values() if r['status'] == 'success'])
            print(f"Average Test Accuracy: {avg_test_acc:.4f}")
        
        print("="*80)
        
        with open('stage2_training_results.json', 'w') as f:
            import json
            json.dump(self.results, f, indent=2)
        
        logger.info("Results saved to stage2_training_results.json")


if __name__ == '__main__':
    symbols_timeframes = [
        ('BTCUSDT', '15m'),
        ('BTCUSDT', '1h'),
        ('ETHUSDT', '15m'),
        ('ETHUSDT', '1h'),
    ]
    
    batch_trainer = Stage2BatchTrainer()
    
    logger.info("Step 1: Preparing data...")
    batch_trainer.prepare_all_data(symbols_timeframes)
    
    logger.info("\nStep 2: Training models...")
    batch_trainer.train_all(symbols_timeframes, cv_folds=5)
