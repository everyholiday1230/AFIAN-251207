"""
Ensemble Signal Generator
=========================

Combines multiple models (XGBoost, TabNet, CatBoost) for improved predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.models.layer3_signal.signal_generator import SignalGenerator
from src.utils.logger import get_logger

logger = get_logger("ensemble_generator")


class EnsembleSignalGenerator:
    """
    Ensemble of multiple signal generators.
    
    Combines predictions from:
    - XGBoost (proven, fast, interpretable)
    - TabNet (deep learning for tabular data)
    - CatBoost (gradient boosting with categorical features support)
    """
    
    def __init__(self, use_gpu: bool = False, confidence_threshold: float = 0.65):
        """
        Initialize ensemble signal generator.
        
        Args:
            use_gpu: Use GPU acceleration if available
            confidence_threshold: Minimum confidence level for predictions (0.0-1.0)
        """
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.models = {}
        self.feature_names = []
        self.is_trained = False
        
        logger.info("Ensemble Signal Generator initialized")
        logger.info(f"  Confidence threshold: {confidence_threshold:.2%}")
        logger.info(f"  TabNet available: {TABNET_AVAILABLE}")
        logger.info(f"  CatBoost available: {CATBOOST_AVAILABLE}")
        logger.info(f"  PyTorch available: {TORCH_AVAILABLE}")
    
    def train(
        self,
        df: pd.DataFrame,
        label_col: str = 'tb_label',
        test_size: float = 0.2,
        balance_method: Optional[str] = 'class_weight',
        **kwargs
    ) -> Dict:
        """
        Train all models in ensemble.
        
        Args:
            df: DataFrame with features and labels
            label_col: Target column name
            test_size: Test set size
            balance_method: 'class_weight', 'undersample', or None
            **kwargs: Additional model parameters
        
        Returns:
            Dictionary with training metrics for each model
        """
        logger.info("\n" + "=" * 80)
        logger.info("🚀 Training Ensemble Model (XGBoost + TabNet + CatBoost)")
        logger.info("=" * 80)
        
        all_metrics = {}
        
        # Train XGBoost
        logger.info("\n[1/3] Training XGBoost...")
        xgb_gen = SignalGenerator(model_type="xgboost", use_gpu=self.use_gpu, confidence_threshold=self.confidence_threshold)
        xgb_metrics = xgb_gen.train(df, label_col, test_size, balance_method, **kwargs)
        self.models['xgboost'] = xgb_gen
        all_metrics['xgboost'] = xgb_metrics
        self.feature_names = xgb_gen.feature_names
        
        # Train TabNet
        if TABNET_AVAILABLE:
            logger.info("\n[2/3] Training TabNet...")
            try:
                from src.models.layer3_signal.tabnet_wrapper import TabNetWrapper
                tabnet_gen = TabNetWrapper(use_gpu=self.use_gpu)
                tabnet_gen.feature_names = self.feature_names
                tabnet_metrics = tabnet_gen.train(df, label_col, test_size, balance_method)
                self.models['tabnet'] = tabnet_gen
                all_metrics['tabnet'] = tabnet_metrics
            except Exception as e:
                logger.warning(f"TabNet training failed: {e}")
                logger.info("Falling back to XGBoost for TabNet slot...")
                tabnet_gen = SignalGenerator(model_type="xgboost", use_gpu=self.use_gpu, confidence_threshold=self.confidence_threshold)
                tabnet_metrics = tabnet_gen.train(df, label_col, test_size, balance_method, **kwargs)
                self.models['tabnet'] = tabnet_gen
                all_metrics['tabnet'] = tabnet_metrics
        else:
            logger.warning("TabNet not available, using XGBoost as fallback...")
            tabnet_gen = SignalGenerator(model_type="xgboost", use_gpu=self.use_gpu)
            tabnet_metrics = tabnet_gen.train(df, label_col, test_size, balance_method, **kwargs)
            self.models['tabnet'] = tabnet_gen
            all_metrics['tabnet'] = tabnet_metrics
        
        # Train CatBoost  
        if CATBOOST_AVAILABLE:
            logger.info("\n[3/3] Training CatBoost...")
            try:
                from src.models.layer3_signal.catboost_wrapper import CatBoostWrapper
                catboost_gen = CatBoostWrapper(use_gpu=self.use_gpu)
                catboost_gen.feature_names = self.feature_names
                catboost_metrics = catboost_gen.train(df, label_col, test_size, balance_method)
                self.models['catboost'] = catboost_gen
                all_metrics['catboost'] = catboost_metrics
            except Exception as e:
                logger.warning(f"CatBoost training failed: {e}")
                logger.info("Falling back to XGBoost for CatBoost slot...")
                catboost_gen = SignalGenerator(model_type="xgboost", use_gpu=self.use_gpu, confidence_threshold=self.confidence_threshold)
                catboost_metrics = catboost_gen.train(df, label_col, test_size, balance_method, **kwargs)
                self.models['catboost'] = catboost_gen
                all_metrics['catboost'] = catboost_metrics
        else:
            logger.warning("CatBoost not available, using XGBoost as fallback...")
            catboost_gen = SignalGenerator(model_type="xgboost", use_gpu=self.use_gpu)
            catboost_metrics = catboost_gen.train(df, label_col, test_size, balance_method, **kwargs)
            self.models['catboost'] = catboost_gen
            all_metrics['catboost'] = catboost_metrics
        
        # Calculate ensemble performance
        logger.info("\n" + "=" * 80)
        logger.info("📊 Model Comparison")
        logger.info("=" * 80)
        
        for model_name, metrics in all_metrics.items():
            if metrics:
                logger.info(f"  {model_name.upper()}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
        
        self.is_trained = True
        
        # Return ensemble metrics
        ensemble_metrics = {
            'individual_metrics': all_metrics,
            'models_trained': list(self.models.keys()),
            'accuracy': all_metrics['xgboost']['accuracy'],  # Use best model's accuracy
            'f1_score': all_metrics['xgboost']['f1_score'],
        }
        
        logger.info("\n✅ Ensemble training completed!")
        
        return ensemble_metrics
    
    def predict(
        self,
        df: pd.DataFrame,
        return_probabilities: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate ensemble predictions using voting.
        
        Args:
            df: DataFrame with features
            return_probabilities: Whether to return probability scores
        
        Returns:
            Tuple of (signals, confidence, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call train() first.")
        
        all_predictions = []
        all_probabilities = []
        
        # Get predictions from each model
        for model_name, model_gen in self.models.items():
            try:
                signals, conf, probs = model_gen.predict(df, return_probabilities=True)
                
                # Convert signals to integers
                signal_mapping = {'LONG': 0, 'SHORT': 1, 'NEUTRAL': 2}
                pred_ints = np.array([signal_mapping[s] for s in signals])
                
                all_predictions.append(pred_ints)
                all_probabilities.append(probs)
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
        
        if len(all_predictions) == 0:
            raise ValueError("No models could make predictions")
        
        # Voting ensemble (majority vote)
        ensemble_pred = stats.mode(np.array(all_predictions), axis=0, keepdims=False)[0]
        
        # Average probabilities
        ensemble_proba = np.mean(all_probabilities, axis=0)
        
        # Decode predictions
        inverse_mapping = {0: 'LONG', 1: 'SHORT', 2: 'NEUTRAL'}
        signals = np.array([inverse_mapping[int(p)] for p in ensemble_pred])
        
        # Get confidence (max probability)
        confidence = np.max(ensemble_proba, axis=1)
        
        # Apply confidence threshold
        low_confidence_mask = confidence < self.confidence_threshold
        signals[low_confidence_mask] = 'NEUTRAL'
        
        if return_probabilities:
            return signals, confidence, ensemble_proba
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
        
        # Predict
        signals, confidence, probabilities = self.predict(df)
        
        # Create probability dictionary
        prob_dict = {
            'LONG': float(probabilities[0][0]),
            'SHORT': float(probabilities[0][1]),
            'NEUTRAL': float(probabilities[0][2])
        }
        
        return signals[0], float(confidence[0]), prob_dict
