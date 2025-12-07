"""
TabNet Wrapper for Signal Generation
=====================================

Wrapper around PyTorch TabNet for trading signal generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger("tabnet_wrapper")


class TabNetWrapper:
    """Wrapper for TabNet model to match SignalGenerator interface."""
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize TabNet wrapper.
        
        Args:
            use_gpu: Use GPU acceleration if available
        """
        if not TABNET_AVAILABLE:
            raise ImportError("pytorch-tabnet not installed. Run: pip install pytorch-tabnet")
        
        self.use_gpu = use_gpu
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.label_mapping = {'LONG': 0, 'SHORT': 1, 'NEUTRAL': 2}
        self.inverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
        logger.info("TabNet Wrapper initialized")
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        label_col: str = 'tb_label',
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from labeled DataFrame."""
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            exclude_cols = [
                'time', 'timestamp', 'symbol', 'timeframe',
                label_col, 'tb_exit_price', 'tb_exit_time', 'tb_return',
                'tb_holding_minutes', 'tb_exit_reason',
                'open', 'high', 'low', 'close', 'volume',
                'regime', 'impulse_color'
            ]
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with insufficient data
        clean_df = df[df['tb_exit_reason'] != 'INSUFFICIENT_DATA'].copy()
        clean_df = clean_df[df['tb_exit_reason'] != 'END_OF_DATA'].copy()
        clean_df = clean_df[feature_cols + [label_col]].dropna()
        
        X = clean_df[feature_cols]
        y = clean_df[label_col]
        
        logger.info(f"Prepared {len(X)} training samples with {len(feature_cols)} features")
        
        return X, y
    
    def train(
        self,
        df: pd.DataFrame,
        label_col: str = 'tb_label',
        test_size: float = 0.2,
        balance_method: Optional[str] = 'class_weight',
        **kwargs
    ) -> Dict:
        """
        Train TabNet model.
        
        Args:
            df: DataFrame with features and labels
            label_col: Target column name
            test_size: Test set size
            balance_method: Not used for TabNet
            **kwargs: Additional model parameters
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training TabNet model...")
        
        # Prepare data
        X, y = self.prepare_training_data(df, label_col)
        if not self.feature_names:
            self.feature_names = list(X.columns)
        
        # Encode labels
        y_encoded = y.map(self.label_mapping).values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # TabNet parameters
        params = {
            'n_d': 8,
            'n_a': 8,
            'n_steps': 3,
            'gamma': 1.3,
            'n_independent': 2,
            'n_shared': 2,
            'lambda_sparse': 1e-3,
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=2e-2),
            'mask_type': 'entmax',
            'scheduler_params': {"step_size": 10, "gamma": 0.9},
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'seed': 42,
            'verbose': 0,  # Reduce verbosity
        }
        
        # Override with user params
        params.update(kwargs)
        
        # Train model
        self.model = TabNetClassifier(**params)
        
        try:
            self.model.fit(
                X_train.values, y_train,
                eval_set=[(X_test.values, y_test)],
                eval_metric=['accuracy'],
                max_epochs=100,  # Reduced for faster training
                patience=15,
                batch_size=256,
            )
        except Exception as e:
            logger.error(f"TabNet training failed: {e}")
            raise
        
        # Predictions
        y_pred = self.model.predict(X_test.values)
        y_pred_proba = self.model.predict_proba(X_test.values)
        
        # Decode predictions
        y_pred_labels = [self.inverse_mapping[p] for p in y_pred]
        y_test_labels = [self.inverse_mapping[p] for p in y_test]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
        
        logger.info(f"✅ TabNet Training completed")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score (weighted): {f1:.4f}")
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'model': self.model
        }
    
    def predict(
        self,
        df: pd.DataFrame,
        return_probabilities: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate trading signals.
        
        Args:
            df: DataFrame with features
            return_probabilities: Whether to return probability scores
        
        Returns:
            Tuple of (signals, confidence, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        available_features = [f for f in self.feature_names if f in df.columns]
        X = df[available_features]
        
        # Predict
        predictions = self.model.predict(X.values)
        probabilities = self.model.predict_proba(X.values)
        
        # Decode predictions
        signals = np.array([self.inverse_mapping[int(p)] for p in predictions])
        
        # Get confidence (max probability)
        confidence = np.max(probabilities, axis=1)
        
        if return_probabilities:
            return signals, confidence, probabilities
        else:
            return signals, confidence, None
