import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


class Stage2Trainer:
    """
    Stage 2 Trainer: Predicts ZigZag signal type (HH/LH/HL/LL)
    
    Input: Features from bars confirmed by Stage 1 as having signals
    Output: Classification into 4 classes
        - 1: HH (Higher High) - Sell signal
        - 2: LH (Lower High) - Sell signal
        - 3: HL (Higher Low) - Buy signal
        - 4: LL (Lower Low) - Buy signal
    """
    
    def __init__(self, model_dir='models/stage2', random_state=42):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.class_names = {1: 'HH', 2: 'LH', 3: 'HL', 4: 'LL'}
        self.class_mapping = {'HH': 1, 'LH': 2, 'HL': 3, 'LL': 4}
        
    def prepare_data(self, X_all, y_all, stage1_predictions):
        """
        Filter data: keep only samples predicted as signal by Stage 1
        
        Parameters:
        -----------
        X_all : ndarray, shape (n_samples, n_features)
            All features from training set
        y_all : ndarray, shape (n_samples,)
            True labels (0/1/2/3/4 where 0 is NO_SIGNAL)
        stage1_predictions : ndarray, shape (n_samples,)
            Stage 1 model predictions (0 = no signal, 1 = has signal)
            
        Returns:
        --------
        X_stage2 : ndarray
            Filtered features (only signal bars)
        y_stage2 : ndarray
            Filtered labels, remapped to 1/2/3/4
        stats : dict
            Statistics about filtering
        """
        
        signal_mask = stage1_predictions == 1
        X_stage2 = X_all[signal_mask]
        y_stage2 = y_all[signal_mask]
        
        # Remove NO_SIGNAL (0) labels if present
        valid_mask = y_stage2 > 0
        X_stage2 = X_stage2[valid_mask]
        y_stage2 = y_stage2[valid_mask]
        
        stats = {
            'total_samples': len(X_all),
            'signal_samples': np.sum(signal_mask),
            'valid_signal_samples': len(X_stage2),
            'signal_percentage': (len(X_stage2) / len(X_all)) * 100,
            'label_distribution': pd.Series(y_stage2).value_counts().to_dict()
        }
        
        print("Data Preparation Summary:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Samples with Stage 1 signal: {stats['signal_samples']}")
        print(f"  Valid Stage 2 samples: {stats['valid_signal_samples']}")
        print(f"  Signal percentage: {stats['signal_percentage']:.2f}%")
        print(f"\n  Label distribution:")
        for label_id in sorted(y_stage2):
            count = (y_stage2 == label_id).sum()
            label_name = self.class_names.get(label_id, 'Unknown')
            pct = (count / len(y_stage2)) * 100
            print(f"    {label_name} ({label_id}): {count} ({pct:.1f}%)")
        
        return X_stage2, y_stage2, stats
    
    def compute_class_weights(self, y):
        """Compute class weights for imbalanced data"""
        unique_classes, class_counts = np.unique(y, return_counts=True)
        total = len(y)
        weights = {}
        for cls, count in zip(unique_classes, class_counts):
            weights[cls] = total / (len(unique_classes) * count)
        return weights
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              normalize=True, cv_folds=5, save_model=True):
        """
        Train Stage 2 classifier
        
        Parameters:
        -----------
        X_train : ndarray, shape (n_train, n_features)
        y_train : ndarray, shape (n_train,)
        X_val : ndarray, optional
        y_val : ndarray, optional
        normalize : bool
            Whether to normalize features using StandardScaler
        cv_folds : int
            Number of cross-validation folds
        save_model : bool
            Whether to save the trained model
            
        Returns:
        --------
        results : dict
            Training results including metrics and model info
        """
        
        self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        if normalize:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val = self.scaler.transform(X_val)
        
        class_weights = self.compute_class_weights(y_train)
        
        print("\nTraining Stage 2 Model...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {len(np.unique(y_train))}")
        print(f"  Class weights: {class_weights}")
        
        if X_val is not None and y_val is not None:
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            params = {
                'objective': 'multiclass',
                'num_class': 4,
                'metric': 'multi_logloss',
                'num_leaves': 31,
                'max_depth': 7,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbose': -1,
                'class_weight': 'balanced'
            }
            
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'validation'],
                callbacks=[
                    lgb.log_evaluation(period=20),
                    lgb.early_stopping(stopping_rounds=20)
                ]
            )
            
            y_pred_val = self.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_pred_val)
            val_f1 = f1_score(y_val, y_pred_val, average='weighted')
            
            print(f"\n  Validation Accuracy: {val_accuracy:.4f}")
            print(f"  Validation F1-Score: {val_f1:.4f}")
        else:
            train_data = lgb.Dataset(X_train, label=y_train)
            
            params = {
                'objective': 'multiclass',
                'num_class': 4,
                'metric': 'multi_logloss',
                'num_leaves': 31,
                'max_depth': 7,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbose': -1,
                'class_weight': 'balanced'
            }
            
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                callbacks=[lgb.log_evaluation(period=50)]
            )
        
        y_pred_train = self.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        
        print(f"\n  Training Accuracy: {train_accuracy:.4f}")
        
        if save_model:
            self.save_model()
        
        results = {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy if X_val is not None else None,
            'feature_count': X_train.shape[1],
            'sample_count': len(X_train),
            'class_weights': class_weights
        }
        
        return results
    
    def predict(self, X):
        """
        Predict class for input features
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
        
        Returns:
        --------
        predictions : ndarray, shape (n_samples,)
            Predicted class labels (1/2/3/4)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        probs = self.model.predict(X)
        predictions = np.argmax(probs, axis=1) + 1
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
        
        Returns:
        --------
        probs : ndarray, shape (n_samples, 4)
            Probability for each class
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        probs = self.model.predict(X)
        return probs
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Parameters:
        -----------
        X_test : ndarray, shape (n_test, n_features)
        y_test : ndarray, shape (n_test,)
        
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print("\nStage 2 Model Evaluation:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        print("\n  Per-Class Metrics:")
        report = classification_report(y_test, y_pred, 
                                       target_names=['HH', 'LH', 'HL', 'LL'],
                                       zero_division=0)
        print(report)
        
        print("\n  Confusion Matrix:")
        print(cm)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
        y : ndarray, shape (n_samples,)
        cv : int
            Number of folds
            
        Returns:
        --------
        cv_results : dict
            Cross-validation results
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=False, random_state=self.random_state)
        
        scores_accuracy = []
        scores_f1 = []
        
        print(f"\nPerforming {cv}-Fold Cross-Validation...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_cv = X[train_idx]
            y_train_cv = y[train_idx]
            X_val_cv = X[val_idx]
            y_val_cv = y[val_idx]
            
            scaler_cv = StandardScaler()
            X_train_cv = scaler_cv.fit_transform(X_train_cv)
            X_val_cv = scaler_cv.transform(X_val_cv)
            
            train_data = lgb.Dataset(X_train_cv, label=y_train_cv)
            val_data = lgb.Dataset(X_val_cv, label=y_val_cv, reference=train_data)
            
            params = {
                'objective': 'multiclass',
                'num_class': 4,
                'metric': 'multi_logloss',
                'num_leaves': 31,
                'max_depth': 7,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbose': -1,
                'class_weight': 'balanced'
            }
            
            model_cv = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[train_data, val_data],
                callbacks=[
                    lgb.log_evaluation(period=0),
                    lgb.early_stopping(stopping_rounds=20)
                ]
            )
            
            y_pred_cv = np.argmax(model_cv.predict(X_val_cv), axis=1) + 1
            
            acc = accuracy_score(y_val_cv, y_pred_cv)
            f1 = f1_score(y_val_cv, y_pred_cv, average='weighted')
            
            scores_accuracy.append(acc)
            scores_f1.append(f1)
            
            print(f"  Fold {fold + 1}: Accuracy={acc:.4f}, F1={f1:.4f}")
        
        mean_acc = np.mean(scores_accuracy)
        std_acc = np.std(scores_accuracy)
        mean_f1 = np.mean(scores_f1)
        std_f1 = np.std(scores_f1)
        
        print(f"\n  Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
        print(f"  Mean F1-Score: {mean_f1:.4f} (+/- {std_f1:.4f})")
        
        cv_results = {
            'accuracy_scores': scores_accuracy,
            'f1_scores': scores_f1,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'mean_f1': mean_f1,
            'std_f1': std_f1
        }
        
        return cv_results
    
    def save_model(self, symbol_timeframe='btcusdt_15m'):
        """Save model and scaler to disk"""
        model_path = self.model_dir / f'{symbol_timeframe}_stage2_model.pkl'
        scaler_path = self.model_dir / f'{symbol_timeframe}_stage2_scaler.pkl'
        
        self.model.save_model(str(model_path).replace('.pkl', '.txt'))
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, symbol_timeframe='btcusdt_15m'):
        """Load model and scaler from disk"""
        model_path = self.model_dir / f'{symbol_timeframe}_stage2_model.txt'
        scaler_path = self.model_dir / f'{symbol_timeframe}_stage2_scaler.pkl'
        
        self.model = lgb.Booster(model_file=str(model_path))
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")
