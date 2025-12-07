"""
Layer 4: Dynamic Position Sizing
=================================

Calculate optimal position size based on:
- Signal confidence
- Market volatility
- Account balance
- Risk parameters

Methods:
- Fixed Percentage
- Kelly Criterion
- Volatility-based
- Confidence-based
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger("position_sizer")


class SizingMethod(Enum):
    """Position sizing methods."""
    FIXED = "FIXED"
    KELLY = "KELLY"
    VOLATILITY = "VOLATILITY"
    CONFIDENCE = "CONFIDENCE"
    HYBRID = "HYBRID"


class PositionSizer:
    """
    Calculate position size dynamically based on multiple factors.
    
    Features:
    - Multiple sizing methods
    - Risk-adjusted sizing
    - Max position limits
    - Volatility adjustment
    """
    
    def __init__(
        self,
        method: str = "HYBRID",
        base_risk_pct: float = 0.02,
        max_position_pct: float = 0.10,
        min_position_pct: float = 0.01,
        confidence_multiplier: float = 2.0,
        volatility_adjustment: bool = True
    ):
        """
        Initialize position sizer.
        
        Args:
            method: Sizing method (FIXED, KELLY, VOLATILITY, CONFIDENCE, HYBRID)
            base_risk_pct: Base risk per trade (e.g., 0.02 = 2%)
            max_position_pct: Maximum position size (e.g., 0.10 = 10%)
            min_position_pct: Minimum position size (e.g., 0.01 = 1%)
            confidence_multiplier: How much confidence affects size
            volatility_adjustment: Adjust for volatility
        """
        self.method = SizingMethod[method.upper()]
        self.base_risk_pct = base_risk_pct
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.confidence_multiplier = confidence_multiplier
        self.volatility_adjustment = volatility_adjustment
        
        logger.info(f"PositionSizer initialized")
        logger.info(f"  Method: {self.method.value}")
        logger.info(f"  Base risk: {base_risk_pct:.2%}")
        logger.info(f"  Max position: {max_position_pct:.2%}")
        logger.info(f"  Min position: {min_position_pct:.2%}")
    
    def calculate_position_size(
        self,
        account_balance: float,
        signal_confidence: float,
        current_price: float,
        stop_loss_pct: float = 0.005,
        volatility: Optional[float] = None,
        regime: Optional[str] = None
    ) -> Dict:
        """
        Calculate position size.
        
        Args:
            account_balance: Current account balance
            signal_confidence: Signal confidence (0-1)
            current_price: Current market price
            stop_loss_pct: Stop loss percentage
            volatility: Current market volatility (ATR%)
            regime: Market regime (for adjustments)
        
        Returns:
            Dictionary with position size details
        """
        # Base calculation based on method
        if self.method == SizingMethod.FIXED:
            size_pct = self._fixed_size()
        elif self.method == SizingMethod.KELLY:
            size_pct = self._kelly_criterion(signal_confidence, stop_loss_pct)
        elif self.method == SizingMethod.VOLATILITY:
            size_pct = self._volatility_based(volatility)
        elif self.method == SizingMethod.CONFIDENCE:
            size_pct = self._confidence_based(signal_confidence)
        else:  # HYBRID
            size_pct = self._hybrid_size(signal_confidence, volatility, stop_loss_pct)
        
        # Apply regime adjustments
        if regime:
            size_pct = self._adjust_for_regime(size_pct, regime)
        
        # Apply limits
        size_pct = np.clip(size_pct, self.min_position_pct, self.max_position_pct)
        
        # Calculate dollar amounts
        position_value = account_balance * size_pct
        shares = position_value / current_price
        
        result = {
            'position_size_pct': size_pct,
            'position_value': position_value,
            'shares': shares,
            'method': self.method.value,
            'confidence': signal_confidence,
            'volatility': volatility,
            'regime': regime
        }
        
        return result
    
    def _fixed_size(self) -> float:
        """Fixed percentage sizing."""
        return self.base_risk_pct * 4  # 2% risk = 8% position (with 4x leverage potential)
    
    def _kelly_criterion(self, confidence: float, risk_pct: float) -> float:
        """
        Kelly Criterion sizing.
        
        Formula: f = (p * b - q) / b
        where:
        - p = probability of win
        - q = probability of loss
        - b = win/loss ratio
        """
        # Assume symmetric payoff (1:1)
        b = 1.0
        
        # Use confidence as win probability
        p = confidence
        q = 1 - confidence
        
        if p <= q:
            return self.min_position_pct
        
        # Kelly formula
        kelly_pct = (p * b - q) / b
        
        # Use fractional Kelly (25% of full Kelly for safety)
        kelly_pct = kelly_pct * 0.25
        
        return kelly_pct
    
    def _volatility_based(self, volatility: Optional[float]) -> float:
        """Volatility-adjusted sizing."""
        if volatility is None:
            return self.base_risk_pct * 4
        
        # Inverse relationship: higher volatility = smaller position
        # Assume normal volatility is 2%
        normal_vol = 0.02
        vol_adjustment = normal_vol / max(volatility / 100, 0.005)
        
        size = self.base_risk_pct * 4 * vol_adjustment
        
        return size
    
    def _confidence_based(self, confidence: float) -> float:
        """Confidence-based sizing."""
        # Linear scaling with confidence
        size = self.base_risk_pct * 4 * (confidence ** self.confidence_multiplier)
        
        return size
    
    def _hybrid_size(
        self,
        confidence: float,
        volatility: Optional[float],
        risk_pct: float
    ) -> float:
        """
        Hybrid sizing combining multiple methods.
        
        Weights:
        - 30% Confidence
        - 30% Volatility
        - 40% Kelly
        """
        # Confidence component
        conf_size = self._confidence_based(confidence)
        
        # Volatility component
        vol_size = self._volatility_based(volatility)
        
        # Kelly component
        kelly_size = self._kelly_criterion(confidence, risk_pct)
        
        # Weighted average
        hybrid_size = (
            0.3 * conf_size +
            0.3 * vol_size +
            0.4 * kelly_size
        )
        
        return hybrid_size
    
    def _adjust_for_regime(self, size: float, regime: str) -> float:
        """
        Adjust position size based on market regime.
        
        Rules:
        - BULL + LOW_VOL: Increase size by 20%
        - BEAR + HIGH_VOL: Decrease size by 30%
        - SIDEWAYS: Decrease size by 10%
        """
        adjustments = {
            'BULL_LOW': 1.2,
            'BULL_MEDIUM': 1.1,
            'BULL_HIGH': 1.0,
            'BEAR_LOW': 0.8,
            'BEAR_MEDIUM': 0.7,
            'BEAR_HIGH': 0.6,
            'SIDEWAYS_LOW': 0.9,
            'SIDEWAYS_MEDIUM': 0.85,
            'SIDEWAYS_HIGH': 0.8,
        }
        
        # Extract trend and volatility from regime string
        if '_' in regime:
            parts = regime.split('_')
            trend = parts[0]
            vol = parts[1]
            
            # Create key
            key = f"{trend}_{vol}"
            
            # Apply adjustment
            if key in adjustments:
                size = size * adjustments[key]
        
        return size
    
    def calculate_leverage(
        self,
        position_size_pct: float,
        max_leverage: float = 5.0
    ) -> float:
        """
        Calculate appropriate leverage based on position size.
        
        Args:
            position_size_pct: Position size as percentage of account
            max_leverage: Maximum allowed leverage
        
        Returns:
            Leverage to use (1.0 = no leverage)
        """
        # If position size <= account balance, no leverage needed
        if position_size_pct <= 1.0:
            return 1.0
        
        # Calculate required leverage
        leverage = position_size_pct
        
        # Cap at max leverage
        leverage = min(leverage, max_leverage)
        
        return leverage
    
    def get_risk_metrics(
        self,
        account_balance: float,
        position_value: float,
        stop_loss_pct: float,
        leverage: float = 1.0
    ) -> Dict:
        """
        Calculate risk metrics for a position.
        
        Returns:
            Dictionary with risk metrics
        """
        # Dollar risk
        dollar_risk = position_value * stop_loss_pct * leverage
        
        # Risk as percentage of account
        risk_pct = dollar_risk / account_balance
        
        # Risk-reward ratio (assuming 1.5% profit target vs 0.5% stop loss)
        risk_reward = 0.015 / stop_loss_pct
        
        return {
            'dollar_risk': dollar_risk,
            'risk_pct': risk_pct,
            'risk_reward_ratio': risk_reward,
            'max_loss': -dollar_risk,
            'max_loss_pct': -risk_pct
        }


# Convenience function
def calculate_position(
    account_balance: float,
    signal_confidence: float,
    current_price: float,
    **kwargs
) -> Dict:
    """
    Quick function to calculate position size.
    
    Args:
        account_balance: Current account balance
        signal_confidence: Signal confidence (0-1)
        current_price: Current market price
        **kwargs: Additional parameters for PositionSizer
    
    Returns:
        Position size details
    """
    sizer = PositionSizer(**kwargs)
    return sizer.calculate_position(
        account_balance=account_balance,
        signal_confidence=signal_confidence,
        current_price=current_price
    )


if __name__ == "__main__":
    # Test position sizer
    print("=== Position Sizer Test ===\n")
    
    # Test parameters
    account = 10000
    price = 45000
    confidence = 0.85
    volatility = 2.5  # 2.5% ATR
    
    # Test different methods
    methods = ['FIXED', 'KELLY', 'VOLATILITY', 'CONFIDENCE', 'HYBRID']
    
    for method in methods:
        print(f"\n{method} Method:")
        sizer = PositionSizer(method=method)
        result = sizer.calculate_position(
            account_balance=account,
            signal_confidence=confidence,
            current_price=price,
            volatility=volatility,
            regime='BULL_MEDIUM_MEDIUM'
        )
        
        print(f"  Position: {result['position_size_pct']:.2%} (${result['position_value']:,.2f})")
        print(f"  Shares: {result['shares']:.4f}")
        
        # Risk metrics
        risk = sizer.get_risk_metrics(
            account_balance=account,
            position_value=result['position_value'],
            stop_loss_pct=0.005
        )
        print(f"  Risk: ${risk['dollar_risk']:.2f} ({risk['risk_pct']:.2%})")
    
    print("\n✅ Position sizer test completed!")
