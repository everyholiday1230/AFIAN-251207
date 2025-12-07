"""
Layer 3: Signal Generator
==========================

Generate LONG/SHORT/NEUTRAL signals based on Triple Barrier labels.

This is the CORE of the trading system - it decides what action to take.

Models (Ensemble):
- XGBoost (Phase 1)
- TabNet (Phase 2)
- FT-Transformer (Phase 2)
- CatBoost (Phase 2)

Philosophy: Learn which actions (LONG/SHORT/NEUTRAL) lead to hitting
profit targets rather than predicting price direction.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger("signal_generator")


class TradingSignal(Enum):
    """Trading signals."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class SignalGenerator:
    """
    Generate trading signals using Triple Barrier labels.
    
    Phase 1: XGBoost (proven, interpretable, fast)
    Phase 2: Ensemble with TabNet + FT-Transformer
    """
    
    def __init__(self, model_type: str = "xgboost", use_gpu: bool = False, confidence_threshold: float = None):
        """
        Initialize signal generator.
        
        Args:
            model_type: Type of model ('xgboost', 'tabnet', 'ensemble')
            use_gpu: Use GPU acceleration if available
            confidence_threshold: Minimum confidence threshold (0.0-1.0), defaults to config value
        """
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.class_weights = None
        
        # Signal confidence threshold
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        else:
            self.confidence_threshold = config.model.signal_confidence_threshold
        
        logger.info(f"Signal Generator initialized with {model_type}")
        logger.info(f"Confidence threshold: {self.confidence_threshold:.2%}")
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        label_col: str = 'tb_label',
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from labeled DataFrame.
        
        Args:
            df: DataFrame with features and Triple Barrier labels
            label_col: Label column name
            feature_cols: List of feature columns (None = auto-detect)
        
        Returns:
            Tuple of (features_df, labels_series)
        """
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            exclude_cols = [
                'time', 'timestamp', 'symbol', 'timeframe',
                label_col, 'tb_exit_price', 'tb_exit_time', 'tb_return',
                'tb_holding_minutes', 'tb_exit_reason',
                'open', 'high', 'low', 'close', 'volume',  # Raw OHLCV
                'regime',  # Will be used as categorical feature
                'impulse_color'  # Used only in indicator calculation, not for model training
            ]
            
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with insufficient data or NaN
        clean_df = df[df['tb_exit_reason'] != 'INSUFFICIENT_DATA'].copy()
        clean_df = clean_df[df['tb_exit_reason'] != 'END_OF_DATA'].copy()
        clean_df = clean_df[feature_cols + [label_col]].dropna()
        
        X = clean_df[feature_cols]
        y = clean_df[label_col]
        
        logger.info(f"Prepared {len(X)} training samples with {len(feature_cols)} features")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")
        
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
        Train signal generation model.
        
        Args:
            df: DataFrame with features and labels
            label_col: Target column name
            test_size: Test set size
            balance_method: 'class_weight', 'undersample', or None
            **kwargs: Additional model parameters
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training signal generator...")
        
        # Prepare data
        X, y = self.prepare_training_data(df, label_col)
        self.feature_names = list(X.columns)
        
        # Calculate class weights if needed
        if balance_method == 'class_weight':
            from sklearn.utils.class_weight import compute_class_weight
            
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            self.class_weights = dict(zip(classes, weights))
            
            logger.info(f"Class weights: {self.class_weights}")
        
        # Split data (stratified to maintain label distribution)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Train model based on type
        if self.model_type == "xgboost":
            return self._train_xgboost(X_train, X_test, y_train, y_test, **kwargs)
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented yet")
    
    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        **kwargs
    ) -> Dict:
        """Train XGBoost model."""
        # Encode labels to integers
        label_mapping = {'LONG': 0, 'SHORT': 1, 'NEUTRAL': 2}
        y_train_encoded = y_train.map(label_mapping)
        y_test_encoded = y_test.map(label_mapping)
        
        # Calculate sample weights
        sample_weights = None
        if self.class_weights:
            sample_weights = np.array([self.class_weights[label] for label in y_train])
        
        # XGBoost parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
            'predictor': 'gpu_predictor' if self.use_gpu else 'cpu_predictor',
            'verbosity': 1,
        }
        
        # Override with user params
        params.update(kwargs)
        
        # Train model (eval_metric already in params)
        self.model = xgb.XGBClassifier(**params)
        
        self.model.fit(
            X_train,
            y_train_encoded,
            sample_weight=sample_weights,
            eval_set=[(X_train, y_train_encoded), (X_test, y_test_encoded)],
            verbose=50
        )
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Decode predictions
        inverse_mapping = {v: k for k, v in label_mapping.items()}
        y_pred_labels = [inverse_mapping[p] for p in y_pred]
        y_test_labels = [inverse_mapping[p] for p in y_test_encoded]
        
        # Metrics
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
        
        # Detailed metrics
        report = classification_report(
            y_test_labels,
            y_pred_labels,
            target_names=['LONG', 'SHORT', 'NEUTRAL'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=['LONG', 'SHORT', 'NEUTRAL'])
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate confidence-filtered accuracy
        confidence_acc = self._calculate_confidence_accuracy(
            y_test_labels, y_pred_labels, y_pred_proba
        )
        
        # Log results
        logger.info(f"✅ Training completed")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score (weighted): {f1:.4f}")
        logger.info(f"Confidence-filtered accuracy: {confidence_acc['accuracy']:.4f} "
                   f"({confidence_acc['samples_kept']}/{len(y_test)} samples kept)")
        
        logger.info("\nPer-class metrics:")
        for signal in ['LONG', 'SHORT', 'NEUTRAL']:
            metrics = report[signal]
            logger.info(f"  {signal}: Precision={metrics['precision']:.3f}, "
                       f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        logger.info("\nTop 10 important features:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'confidence_accuracy': confidence_acc,
            'model': self.model
        }
    
    def _calculate_confidence_accuracy(
        self,
        y_true: List[str],
        y_pred: List[str],
        y_proba: np.ndarray
    ) -> Dict:
        """Calculate accuracy after filtering by confidence threshold."""
        # Get max probability for each prediction
        max_proba = np.max(y_proba, axis=1)
        
        # Filter by confidence
        confident_mask = max_proba >= self.confidence_threshold
        
        if confident_mask.sum() > 0:
            y_true_confident = [y for y, m in zip(y_true, confident_mask) if m]
            y_pred_confident = [y for y, m in zip(y_pred, confident_mask) if m]
            
            confidence_accuracy = accuracy_score(y_true_confident, y_pred_confident)
        else:
            confidence_accuracy = 0.0
        
        return {
            'accuracy': confidence_accuracy,
            'samples_kept': confident_mask.sum(),
            'samples_total': len(y_true),
            'percentage_kept': confident_mask.mean()
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
        
        # Prepare features (use trained feature names, or infer from input)
        if hasattr(self, 'feature_names') and self.feature_names is not None:
            # Use saved feature names from training
            available_features = [f for f in self.feature_names if f in df.columns]
            X = df[available_features]
        else:
            # Fallback: use all numeric columns
            X = df.select_dtypes(include=[np.number])
            self.feature_names = X.columns.tolist()
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Decode predictions
        label_mapping = {0: 'LONG', 1: 'SHORT', 2: 'NEUTRAL'}
        signals = np.array([label_mapping[p] for p in predictions])
        
        # Get confidence (max probability)
        confidence = np.max(probabilities, axis=1)
        
        # Apply confidence threshold
        signals = np.where(
            confidence >= self.confidence_threshold,
            signals,
            'NEUTRAL'  # Default to NEUTRAL if not confident
        )
        
        if return_probabilities:
            return signals, confidence, probabilities
        else:
            return signals, confidence, None
    
    def predict_single(
        self,
        features: Dict,
        regime: Optional[str] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Generate signal for a single data point.
        
        Args:
            features: Dictionary of features
            regime: Optional market regime context
        
        Returns:
            Tuple of (signal, confidence, probabilities_dict)
        """
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0  # Default value
        
        # Predict
        signals, confidence, probabilities = self.predict(df)
        
        # Create probability dictionary
        prob_dict = {
            'LONG': float(probabilities[0][0]),
            'SHORT': float(probabilities[0][1]),
            'NEUTRAL': float(probabilities[0][2])
        }
        
        logger.debug(
            f"Signal: {signals[0]} (confidence: {confidence[0]:.2%}) | "
            f"Regime: {regime} | "
            f"Probs: L={prob_dict['LONG']:.2%}, S={prob_dict['SHORT']:.2%}, N={prob_dict['NEUTRAL']:.2%}"
        )
        
        return signals[0], float(confidence[0]), prob_dict
    
    def save(self, filepath: str):
        """Save model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'class_weights': self.class_weights,
            'confidence_threshold': self.confidence_threshold,
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.class_weights = model_data.get('class_weights')
        self.confidence_threshold = model_data.get('confidence_threshold', 0.65)
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


# Convenience function for quick signal generation
def generate_signal(
    features: Dict,
    model_path: str,
    regime: Optional[str] = None
) -> Tuple[str, float]:
    """
    Quick function to generate a trading signal.
    
    Args:
        features: Feature dictionary
        model_path: Path to trained model
        regime: Optional market regime
    
    Returns:
        Tuple of (signal, confidence)
    """
    generator = SignalGenerator()
    generator.load(model_path)
    
    signal, confidence, probs = generator.predict_single(features, regime)
    
    return signal, confidence


if __name__ == "__main__":
    # Test signal generator
    print("=== Signal Generator Test ===\n")
    
    # Create sample data with Triple Barrier labels
    from src.data_processing.triple_barrier import TripleBarrierLabeler
    from src.data_processing.feature_engineer import FeatureEngineer
    
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=2000, freq='1min')
    
    # Simulate realistic price movements
    returns = np.random.randn(2000) * 0.001 + 0.0001
    prices = 45000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'time': dates,
        'open': prices + np.random.randn(2000) * 10,
        'high': prices + abs(np.random.randn(2000) * 50),
        'low': prices - abs(np.random.randn(2000) * 50),
        'close': prices,
        'volume': np.random.rand(2000) * 100 + 50,
    })
    
    print(f"Sample data shape: {df.shape}")
    
    # Create features
    print("\nCreating features...")
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    print(f"Features created: {len(df_features.columns)}")
    
    # Create Triple Barrier labels
    print("\nCreating Triple Barrier labels...")
    labeler = TripleBarrierLabeler()
    df_labeled = labeler.create_labels(df_features)
    
    # Train signal generator
    print("\nTraining signal generator...")
    generator = SignalGenerator(model_type="xgboost")
    
    metrics = generator.train(df_labeled, test_size=0.2)
    
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Test prediction
    print("\nTesting prediction...")
    test_sample = df_features.tail(10)
    signals, confidence, probs = generator.predict(test_sample)
    
    print("\nPredictions:")
    for i, (signal, conf) in enumerate(zip(signals, confidence)):
        print(f"  Sample {i+1}: {signal} (confidence: {conf:.2%})")
    
    print("\n✅ Signal generator test completed!")
