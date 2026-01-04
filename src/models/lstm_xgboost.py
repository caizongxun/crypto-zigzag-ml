import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class LSTMXGBoostModel:
    """
    Hybrid LSTM + LightGBM model for zigzag signal classification.
    """
    
    def __init__(self, timesteps: int = 60, n_features: int = 55,
                 lstm_units: List[int] = [128, 64], dropout: float = 0.3):
        """
        Initialize model architecture.
        
        Args:
            timesteps: Number of time steps for LSTM
            n_features: Number of input features
            lstm_units: List of LSTM layer units
            dropout: Dropout rate for regularization
        """
        self.timesteps = timesteps
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.scaler = StandardScaler()
        self.lstm_model = None
        self.gbm_model = None
        self.feature_scaler = StandardScaler()
    
    def build_lstm_encoder(self) -> keras.Model:
        """
        Build LSTM encoder to extract temporal patterns.
        
        Returns:
            Compiled LSTM model
        """
        model = models.Sequential([
            layers.LSTM(self.lstm_units[0], 
                       return_sequences=True,
                       input_shape=(self.timesteps, self.n_features),
                       activation='relu'),
            layers.Dropout(self.dropout),
            
            layers.LSTM(self.lstm_units[1], 
                       return_sequences=False,
                       activation='relu'),
            layers.Dropout(self.dropout),
            
            layers.Dense(32, activation='relu')
        ])
        
        return model
    
    def create_sequences(self, data: np.ndarray, labels: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.
        
        Args:
            data: Feature array (n_samples, n_features)
            labels: Label array (optional)
            
        Returns:
            X: Sequences (n_sequences, timesteps, n_features)
            y: Labels (optional)
        """
        X = []
        y_seq = []
        
        for i in range(len(data) - self.timesteps):
            X.append(data[i:i + self.timesteps])
            if labels is not None:
                y_seq.append(labels[i + self.timesteps])
        
        X = np.array(X)
        y_seq = np.array(y_seq) if labels is not None else None
        
        return X, y_seq
    
    def extract_lstm_features(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Extract features using trained LSTM encoder.
        
        Args:
            X_seq: Sequence data
            
        Returns:
            Extracted features
        """
        if self.lstm_model is None:
            raise ValueError('LSTM model not trained yet')
        
        # Use LSTM to extract features (without output layer)
        lstm_feature_extractor = models.Model(
            inputs=self.lstm_model.input,
            outputs=self.lstm_model.get_layer(index=-1).input
        )
        features = lstm_feature_extractor.predict(X_seq, verbose=0)
        return features
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 100, batch_size: int = 32,
             early_stopping_patience: int = 10) -> Dict:
        """
        Train the complete hybrid model.
        
        Args:
            X_train: Training sequences (n_samples, timesteps, n_features)
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training history
        """
        # Build and train LSTM encoder
        print('Training LSTM encoder...')
        self.lstm_model = self.build_lstm_encoder()
        self.lstm_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Prepare target for LSTM (we'll use it as reconstruction task)
        # Actually use for sequence classification
        lstm_model_classifier = models.Sequential([
            layers.LSTM(self.lstm_units[0], 
                       return_sequences=True,
                       input_shape=(self.timesteps, self.n_features),
                       activation='relu'),
            layers.Dropout(self.dropout),
            layers.LSTM(self.lstm_units[1], 
                       return_sequences=False,
                       activation='relu'),
            layers.Dropout(self.dropout),
            layers.Dense(32, activation='relu'),
            layers.Dense(5, activation='softmax')  # 5 classes
        ])
        
        lstm_model_classifier.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        )
        
        history_lstm = lstm_model_classifier.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Save LSTM for feature extraction
        self.lstm_model = models.Model(
            inputs=lstm_model_classifier.input,
            outputs=lstm_model_classifier.get_layer(index=-2).output
        )
        
        # Extract features from both train and validation
        print('Extracting LSTM features for GBM training...')
        lstm_train_features = self.lstm_model.predict(X_train, verbose=0)
        lstm_val_features = self.lstm_model.predict(X_val, verbose=0)
        
        # Train LightGBM on combined features
        print('Training LightGBM model...')
        
        # Combine LSTM features with original features (use last timestep)
        X_train_combined = np.hstack([
            lstm_train_features,
            X_train[:, -1, :]  # Last timestep features
        ])
        
        X_val_combined = np.hstack([
            lstm_val_features,
            X_val[:, -1, :]
        ])
        
        # LightGBM model
        self.gbm_model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1
        )
        
        self.gbm_model.fit(
            X_train_combined, y_train,
            eval_set=[(X_val_combined, y_val)],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        return {
            'lstm_history': history_lstm,
            'lstm_train_acc': history_lstm.history['accuracy'][-1],
            'lstm_val_acc': history_lstm.history['val_accuracy'][-1]
        }
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on test data.
        
        Args:
            X_test: Test sequences
            
        Returns:
            Predictions and probabilities
        """
        if self.lstm_model is None or self.gbm_model is None:
            raise ValueError('Model not trained yet')
        
        # Extract LSTM features
        lstm_features = self.lstm_model.predict(X_test, verbose=0)
        
        # Combine with original features
        X_test_combined = np.hstack([
            lstm_features,
            X_test[:, -1, :]
        ])
        
        # GBM prediction
        predictions = self.gbm_model.predict(X_test_combined)
        probabilities = self.gbm_model.predict_proba(X_test_combined)
        
        return predictions, probabilities
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test sequences
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        predictions, _ = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
            'f1': f1_score(y_test, predictions, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, predictions)
        }
        
        return metrics
