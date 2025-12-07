"""
Layer 1: Market Regime Classifier
==================================

Identifies the current market regime to apply appropriate trading strategies.

Market Regimes:
1. TRENDING_UP - Strong uptrend
2. TRENDING_DOWN - Strong downtrend
3. RANGING - Sideways/choppy market
4. HIGH_VOLATILITY - Increased volatility
5. LOW_VOLATILITY - Low volatility
6. BREAKOUT - Price breaking key levels
7. REVERSAL - Potential trend reversal

Uses: LightGBM (Phase 1) → TFT (Phase 2)
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger("regime_classifier")


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"


class RegimeClassifier:
    """
    Classify market regime using price action and volatility metrics.
    
    Phase 1: LightGBM (simple, fast, interpretable)
    Phase 2: Temporal Fusion Transformer (advanced sequence modeling)
    """
    
    def __init__(self, model_type: str = "lightgbm"):
        """
        Initialize regime classifier.
        
        Args:
            model_type: Type of model ('lightgbm' or 'tft')
        """
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
        
        # Thresholds for regime classification
        self.trend_threshold = 0.02  # 2% price movement for trend
        self.volatility_high = 0.03  # 3% volatility for high vol regime
        self.volatility_low = 0.01  # 1% volatility for low vol regime
        
        logger.info(f"Regime Classifier initialized with {model_type}")
    
    def create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for regime classification.
        
        Args:
            df: DataFrame with OHLCV and technical indicators
        
        Returns:
            DataFrame with regime-specific features
        """
        result = df.copy()
        
        # Trend indicators
        for period in [10, 20, 50]:
            # Price trend
            result[f'trend_{period}'] = result['close'].pct_change(period)
            
            # Moving average slope
            ma = result['close'].rolling(period).mean()
            result[f'ma_slope_{period}'] = ma.pct_change(5)
        
        # Volatility indicators
        for period in [10, 20, 50]:
            returns = result['close'].pct_change()
            result[f'volatility_{period}'] = returns.rolling(period).std()
        
        # ADX for trend strength (if available)
        if 'adx' in result.columns:
            result['trend_strength'] = result['adx']
        
        # Bollinger Band width (volatility proxy)
        if 'bb_width' in result.columns:
            result['bb_width_ma'] = result['bb_width'].rolling(20).mean()
            result['bb_expansion'] = result['bb_width'] / result['bb_width_ma']
        
        # Price position in range
        for period in [20, 50]:
            high_period = result['high'].rolling(period).max()
            low_period = result['low'].rolling(period).min()
            range_period = high_period - low_period
            result[f'price_position_{period}'] = (
                (result['close'] - low_period) / range_period
            ).fillna(0.5)
        
        # Volume trend
        if 'volume' in result.columns:
            result['volume_trend'] = result['volume'].pct_change(10)
        
        # Rate of change (momentum)
        for period in [5, 10, 20]:
            result[f'roc_{period}'] = result['close'].pct_change(period)
        
        # ATR ratio (current ATR / average ATR)
        if 'atr' in result.columns:
            result['atr_ratio'] = result['atr'] / result['atr'].rolling(20).mean()
        
        return result
    
    def label_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create regime labels based on price action and volatility.
        
        Args:
            df: DataFrame with price data and features
        
        Returns:
            DataFrame with regime labels
        """
        result = df.copy()
        regimes = []
        
        for i in range(len(result)):
            # Get current metrics
            trend_20 = result.iloc[i].get('trend_20', 0)
            volatility_20 = result.iloc[i].get('volatility_20', 0.02)
            adx = result.iloc[i].get('adx', 0.5) if 'adx' in result.columns else 0.5
            bb_width = result.iloc[i].get('bb_width', 0.02) if 'bb_width' in result.columns else 0.02
            price_position_20 = result.iloc[i].get('price_position_20', 0.5)
            
            # Determine regime based on multiple factors
            
            # High volatility regime (overrides others)
            if volatility_20 > self.volatility_high:
                regime = MarketRegime.HIGH_VOLATILITY.value
            
            # Low volatility regime
            elif volatility_20 < self.volatility_low and adx < 0.2:
                regime = MarketRegime.LOW_VOLATILITY.value
            
            # Strong uptrend
            elif trend_20 > self.trend_threshold and adx > 0.25:
                # Check for breakout (price near high)
                if price_position_20 > 0.9:
                    regime = MarketRegime.BREAKOUT.value
                else:
                    regime = MarketRegime.TRENDING_UP.value
            
            # Strong downtrend
            elif trend_20 < -self.trend_threshold and adx > 0.25:
                # Check for potential reversal (price near low)
                if price_position_20 < 0.1:
                    regime = MarketRegime.REVERSAL.value
                else:
                    regime = MarketRegime.TRENDING_DOWN.value
            
            # Ranging market (no strong trend)
            else:
                regime = MarketRegime.RANGING.value
            
            regimes.append(regime)
        
        result['regime'] = regimes
        
        # Log regime distribution
        regime_counts = result['regime'].value_counts()
        logger.debug(f"Regime distribution: {regime_counts.to_dict()}")
        
        return result
    
    def train(
        self,
        df: pd.DataFrame,
        target_col: str = 'regime',
        test_size: float = 0.2,
        **kwargs
    ) -> Dict:
        """
        Train regime classifier.
        
        Args:
            df: DataFrame with features and labels
            target_col: Target column name
            test_size: Test set size
            **kwargs: Additional training parameters
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training regime classifier...")
        
        # Prepare features
        exclude_cols = [
            'time', 'timestamp', 'symbol', 'timeframe',
            target_col, 'open', 'high', 'low', 'close', 'volume'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with NaN
        clean_df = df[feature_cols + [target_col]].dropna()
        
        X = clean_df[feature_cols]
        y = clean_df[target_col]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        self.feature_names = feature_cols
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Features: {len(feature_cols)}")
        
        # Train LightGBM
        if self.model_type == "lightgbm":
            # LightGBM parameters
            params = {
                'objective': 'multiclass',
                'num_class': len(self.label_encoder.classes_),
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
            }
            
            # Override with user params
            params.update(kwargs)
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            # Train model
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[train_data, test_data],
                valid_names=['train', 'test'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=50)
                ]
            )
            
            # Predictions
            y_pred = np.argmax(self.model.predict(X_test), axis=1)
            
            # Metrics
            from sklearn.metrics import accuracy_score, classification_report
            
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Test accuracy: {accuracy:.4f}")
            
            # Classification report
            report = classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 10 features:")
            for i, row in feature_importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.0f}")
            
            self.is_trained = True
            
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'feature_importance': feature_importance,
                'model': self.model
            }
        
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented yet")
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict market regime.
        
        Args:
            df: DataFrame with features
        
        Returns:
            Tuple of (regime_labels, confidence_scores)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        X = df[self.feature_names]
        
        # Predict probabilities
        if self.model_type == "lightgbm":
            probs = self.model.predict(X)
            predictions = np.argmax(probs, axis=1)
            confidence = np.max(probs, axis=1)
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented")
        
        # Decode labels
        regime_labels = self.label_encoder.inverse_transform(predictions)
        
        return regime_labels, confidence
    
    def predict_single(self, features: Dict) -> Tuple[str, float]:
        """
        Predict regime for a single data point.
        
        Args:
            features: Dictionary of features
        
        Returns:
            Tuple of (regime, confidence)
        """
        # Create DataFrame from features
        df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0  # Default value
        
        # Predict
        regimes, confidences = self.predict(df)
        
        return regimes[0], confidences[0]
    
    def save(self, filepath: str):
        """Save model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'thresholds': {
                'trend': self.trend_threshold,
                'volatility_high': self.volatility_high,
                'volatility_low': self.volatility_low,
            }
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        
        if 'thresholds' in model_data:
            self.trend_threshold = model_data['thresholds']['trend']
            self.volatility_high = model_data['thresholds']['volatility_high']
            self.volatility_low = model_data['thresholds']['volatility_low']
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test regime classifier
    print("=== Regime Classifier Test ===\n")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1h')
    
    # Simulate different regimes
    prices = []
    current_price = 45000
    
    for i in range(1000):
        if i < 200:  # Uptrend
            change = np.random.randn() * 50 + 20
        elif i < 400:  # Downtrend
            change = np.random.randn() * 50 - 20
        elif i < 600:  # Ranging
            change = np.random.randn() * 30
        elif i < 800:  # High volatility
            change = np.random.randn() * 150
        else:  # Low volatility
            change = np.random.randn() * 20
        
        current_price += change
        prices.append(current_price)
    
    df = pd.DataFrame({
        'time': dates,
        'open': np.array(prices) + np.random.randn(1000) * 20,
        'high': np.array(prices) + abs(np.random.randn(1000) * 50),
        'low': np.array(prices) - abs(np.random.randn(1000) * 50),
        'close': prices,
        'volume': np.random.rand(1000) * 1000 + 500,
    })
    
    print(f"Sample data shape: {df.shape}\n")
    
    # Create classifier
    classifier = RegimeClassifier(model_type="lightgbm")
    
    # Create regime features
    print("Creating regime features...")
    df_features = classifier.create_regime_features(df)
    print(f"Features created: {len(df_features.columns) - len(df.columns)}\n")
    
    # Label regimes
    print("Labeling regimes...")
    df_labeled = classifier.label_regimes(df_features)
    print("\nRegime distribution:")
    print(df_labeled['regime'].value_counts())
    print()
    
    # Train classifier
    print("Training classifier...")
    metrics = classifier.train(df_labeled)
    print(f"\nTest accuracy: {metrics['accuracy']:.4f}")
    
    # Test prediction
    print("\nTesting prediction on new data...")
    test_sample = df_features.tail(10)
    regimes, confidence = classifier.predict(test_sample)
    
    print("\nPredictions:")
    for i, (regime, conf) in enumerate(zip(regimes, confidence)):
        print(f"  Sample {i+1}: {regime} (confidence: {conf:.2%})")
    
    print("\n✅ Regime classifier test completed!")
