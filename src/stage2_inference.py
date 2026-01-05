import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tensorflow import keras


class PipelineInference:
    """
    Two-Stage Inference Pipeline
    
    Stage 1: Signal Detection (Has signal? Yes/No)
    Stage 2: Signal Type Classification (HH/LH/HL/LL)
    """
    
    def __init__(self, stage1_model_path=None, stage2_model_path=None,
                 stage2_scaler_path=None, model_dir='models'):
        self.model_dir = Path(model_dir)
        
        self.stage1_model = None
        self.stage2_model = None
        self.stage2_scaler = None
        
        self.stage1_model_path = stage1_model_path
        self.stage2_model_path = stage2_model_path
        self.stage2_scaler_path = stage2_scaler_path
        
        self.class_names = {1: 'HH', 2: 'LH', 3: 'HL', 4: 'LL'}
        self.signal_types = {'HH': 'SELL', 'LH': 'SELL', 'HL': 'BUY', 'LL': 'BUY'}
        
    def load_stage1_model(self, symbol_timeframe='btcusdt_15m'):
        """
        Load Stage 1 model (signal detection)
        
        Parameters:
        -----------
        symbol_timeframe : str
            e.g., 'btcusdt_15m', 'ethusdt_1h'
        """
        model_path = self.model_dir / f'{symbol_timeframe}' / 'classification.h5'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Stage 1 model not found: {model_path}")
        
        self.stage1_model = keras.models.load_model(str(model_path))
        print(f"Stage 1 model loaded: {model_path}")
        
    def load_stage2_model(self, symbol_timeframe='btcusdt_15m'):
        """
        Load Stage 2 model and scaler (signal type classification)
        
        Parameters:
        -----------
        symbol_timeframe : str
            e.g., 'btcusdt_15m', 'ethusdt_1h'
        """
        model_path = self.model_dir / 'stage2' / f'{symbol_timeframe}_stage2_model.txt'
        scaler_path = self.model_dir / 'stage2' / f'{symbol_timeframe}_stage2_scaler.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Stage 2 model not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Stage 2 scaler not found: {scaler_path}")
        
        import lightgbm as lgb
        
        self.stage2_model = lgb.Booster(model_file=str(model_path))
        
        with open(scaler_path, 'rb') as f:
            self.stage2_scaler = pickle.load(f)
        
        print(f"Stage 2 model loaded: {model_path}")
        print(f"Stage 2 scaler loaded: {scaler_path}")
        
    def predict(self, X_features, return_confidence=True):
        """
        Two-stage prediction
        
        Parameters:
        -----------
        X_features : ndarray, shape (n_samples, n_features)
            Feature matrix
        return_confidence : bool
            Whether to return confidence scores
            
        Returns:
        --------
        predictions : dict
            Contains:
            - 'stage1_signal': ndarray, Stage 1 predictions (0/1)
            - 'stage2_type': ndarray, Stage 2 predictions (1/2/3/4)
            - 'stage2_type_name': list, Readable signal types (HH/LH/HL/LL)
            - 'action': list, Trading action (BUY/SELL/NONE)
            - 'confidence': dict, Confidence scores if return_confidence=True
        """
        
        if self.stage1_model is None:
            raise ValueError("Stage 1 model not loaded")
        
        stage1_probs = self.stage1_model.predict(X_features, verbose=0)
        stage1_preds = (stage1_probs[:, 1] > 0.5).astype(int)
        
        stage2_preds = np.zeros(len(X_features), dtype=int)
        stage2_type_names = []
        actions = []
        
        stage1_confidence = np.max(stage1_probs, axis=1)
        stage2_confidence = np.zeros(len(X_features))
        
        if self.stage2_model is not None:
            signal_mask = stage1_preds == 1
            
            if signal_mask.sum() > 0:
                X_signal = X_features[signal_mask]
                X_signal_scaled = self.stage2_scaler.transform(X_signal)
                
                stage2_probs = self.stage2_model.predict(X_signal_scaled)
                stage2_preds_signal = np.argmax(stage2_probs, axis=1) + 1
                stage2_confidence_signal = np.max(stage2_probs, axis=1)
                
                stage2_preds[signal_mask] = stage2_preds_signal
                stage2_confidence[signal_mask] = stage2_confidence_signal
        
        for i, (s1, s2) in enumerate(zip(stage1_preds, stage2_preds)):
            if s1 == 0:
                stage2_type_names.append('NONE')
                actions.append('NONE')
            else:
                type_name = self.class_names.get(s2, 'UNKNOWN')
                stage2_type_names.append(type_name)
                action = self.signal_types.get(type_name, 'NONE')
                actions.append(action)
        
        predictions = {
            'stage1_signal': stage1_preds,
            'stage2_type': stage2_preds,
            'stage2_type_name': stage2_type_names,
            'action': actions
        }
        
        if return_confidence:
            predictions['confidence'] = {
                'stage1': stage1_confidence,
                'stage2': stage2_confidence,
                'combined': (stage1_confidence * 0.4 + stage2_confidence * 0.6) 
                            if self.stage2_model else stage1_confidence
            }
        
        return predictions
    
    def predict_single(self, X_single, return_confidence=True):
        """
        Predict for a single sample
        
        Parameters:
        -----------
        X_single : ndarray, shape (1, n_features)
            Single feature vector
        return_confidence : bool
        
        Returns:
        --------
        result : dict
            Prediction result
        """
        predictions = self.predict(X_single.reshape(1, -1), return_confidence)
        
        result = {
            'has_signal': bool(predictions['stage1_signal'][0]),
            'signal_type': predictions['stage2_type_name'][0],
            'action': predictions['action'][0]
        }
        
        if return_confidence:
            result['stage1_confidence'] = float(predictions['confidence']['stage1'][0])
            result['stage2_confidence'] = float(predictions['confidence']['stage2'][0])
            result['combined_confidence'] = float(predictions['confidence']['combined'][0])
        
        return result
    
    def get_summary_stats(self, predictions):
        """
        Get summary statistics for batch predictions
        
        Parameters:
        -----------
        predictions : dict
            Output from self.predict()
            
        Returns:
        --------
        stats : dict
            Summary statistics
        """
        stage1_preds = predictions['stage1_signal']
        stage2_preds = predictions['stage2_type']
        actions = predictions['action']
        
        total = len(stage1_preds)
        signal_count = (stage1_preds == 1).sum()
        no_signal_count = (stage1_preds == 0).sum()
        
        buy_count = (np.array(actions) == 'BUY').sum()
        sell_count = (np.array(actions) == 'SELL').sum()
        none_count = (np.array(actions) == 'NONE').sum()
        
        type_distribution = {}
        for i in [1, 2, 3, 4]:
            type_distribution[self.class_names[i]] = (stage2_preds == i).sum()
        
        stats = {
            'total_samples': total,
            'stage1_signal': signal_count,
            'stage1_no_signal': no_signal_count,
            'signal_percentage': (signal_count / total) * 100 if total > 0 else 0,
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'no_signals': none_count,
            'type_distribution': type_distribution,
            'average_stage1_confidence': predictions['confidence']['stage1'].mean(),
            'average_stage2_confidence': predictions['confidence']['stage2'].mean(),
            'average_combined_confidence': predictions['confidence']['combined'].mean()
        }
        
        return stats
    
    def print_summary(self, stats):
        """Print formatted summary statistics"""
        print("\n" + "="*60)
        print("PIPELINE INFERENCE SUMMARY")
        print("="*60)
        print(f"\nTotal Samples: {stats['total_samples']}")
        print(f"\nStage 1 (Signal Detection):")
        print(f"  Signals found: {stats['stage1_signal']} ({stats['signal_percentage']:.2f}%)")
        print(f"  No signals: {stats['stage1_no_signal']}")
        print(f"  Avg confidence: {stats['average_stage1_confidence']:.4f}")
        
        print(f"\nStage 2 (Signal Type Classification):")
        print(f"  Type distribution:")
        for signal_type, count in stats['type_distribution'].items():
            pct = (count / stats['total_samples']) * 100
            print(f"    {signal_type}: {count} ({pct:.2f}%)")
        print(f"  Avg confidence: {stats['average_stage2_confidence']:.4f}")
        
        print(f"\nFinal Actions:")
        print(f"  BUY signals: {stats['buy_signals']}")
        print(f"  SELL signals: {stats['sell_signals']}")
        print(f"  No action: {stats['no_signals']}")
        
        print(f"\nCombined Pipeline Confidence: {stats['average_combined_confidence']:.4f}")
        print("="*60 + "\n")
