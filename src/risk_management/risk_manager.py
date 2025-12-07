"""
Comprehensive Risk Management System
=====================================

3-Tier Risk Management:
Level 1: Position Level - Stop loss, take profit, position sizing
Level 2: Daily Level - Daily loss limits, trade limits, consecutive losses
Level 3: System Level - Emergency shutdown, black swan detection

Philosophy: Survival first, profits second.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import config
from src.utils.logger import get_logger, log_risk_event

logger = get_logger("risk_manager")


class RiskLevel(Enum):
    """Risk severity levels."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class ActionType(Enum):
    """Risk management actions."""
    ALLOW = "ALLOW"
    REDUCE_SIZE = "REDUCE_SIZE"
    HALT_NEW_TRADES = "HALT_NEW_TRADES"
    CLOSE_POSITIONS = "CLOSE_POSITIONS"
    EMERGENCY_SHUTDOWN = "EMERGENCY_SHUTDOWN"


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot."""
    timestamp: datetime
    total_equity: float
    open_positions: int
    used_margin: float
    unrealized_pnl: float
    realized_pnl_today: float
    current_drawdown: float
    max_drawdown: float
    daily_trades: int
    consecutive_losses: int
    sharpe_ratio: Optional[float] = None
    win_rate: Optional[float] = None


@dataclass
class RiskDecision:
    """Risk management decision."""
    action: ActionType
    level: RiskLevel
    reason: str
    recommended_position_size: Optional[float] = None
    should_close_positions: bool = False
    should_halt_trading: bool = False


class RiskManager:
    """
    Comprehensive risk management system with 3 tiers of protection.
    
    This is the most critical component for system survival.
    """
    
    def __init__(self):
        # Configuration
        self.config_risk = config.risk
        self.config_trading = config.trading
        
        # Position Level Limits
        self.max_position_size = config.trading.max_position_size
        self.max_leverage = config.trading.max_leverage
        self.stop_loss_pct = config.risk.stop_loss_pct
        self.take_profit_pct = config.risk.take_profit_pct
        
        # Daily Level Limits
        self.max_daily_loss = config.risk.max_daily_loss
        self.max_daily_trades = config.risk.max_daily_trades
        self.max_consecutive_losses = config.risk.max_consecutive_losses
        
        # System Level Limits
        self.max_drawdown = config.risk.max_drawdown
        self.min_sharpe_ratio = config.risk.min_sharpe_ratio
        self.emergency_volatility_multiplier = config.risk.emergency_volatility_multiplier
        
        # State tracking
        self.initial_capital = config.trading.initial_capital
        self.peak_equity = self.initial_capital
        self.daily_start_equity = self.initial_capital
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_reset_date = datetime.now().date()
        
        # Historical data for calculations
        self.equity_history: List[float] = [self.initial_capital]
        self.daily_returns: List[float] = []
        self.trade_results: List[float] = []  # Recent trade P&Ls
        
        # Emergency state
        self.emergency_shutdown = False
        self.trading_halted = False
        self.halt_reason = None
        
        logger.info("🛡️ Risk Manager initialized")
        logger.info(f"  Position limits: {self.max_position_size:.1%} max size, {self.max_leverage}x max leverage")
        logger.info(f"  Daily limits: {self.max_daily_loss:.1%} max loss, {self.max_daily_trades} max trades")
        logger.info(f"  System limits: {self.max_drawdown:.1%} max drawdown")
    
    # ========================================================================
    # Level 1: Position-Level Risk Management
    # ========================================================================
    
    def calculate_position_size(
        self,
        equity: float,
        signal_confidence: float,
        current_volatility: float,
        avg_volatility: float
    ) -> float:
        """
        Calculate appropriate position size based on confidence and volatility.
        
        Args:
            equity: Current account equity
            signal_confidence: Model confidence (0-1)
            current_volatility: Current market volatility
            avg_volatility: Average historical volatility
        
        Returns:
            Position size as fraction of equity
        """
        # Base position size
        base_size = self.max_position_size
        
        # Adjust for confidence (0.5-1.0 confidence → 0.5-1.0 size multiplier)
        confidence_multiplier = max(0.5, min(1.0, signal_confidence))
        
        # Adjust for volatility (reduce size in high volatility)
        if avg_volatility > 0:
            vol_ratio = current_volatility / avg_volatility
            vol_multiplier = 1.0 / (1 + max(0, vol_ratio - 1))  # Reduce if volatility is high
        else:
            vol_multiplier = 1.0
        
        # Adjust for current drawdown (reduce size during drawdown)
        current_dd = self.calculate_current_drawdown(equity)
        dd_multiplier = max(0.5, 1.0 - (current_dd / self.max_drawdown))
        
        # Calculate final position size
        position_size = base_size * confidence_multiplier * vol_multiplier * dd_multiplier
        
        # Ensure within limits
        position_size = max(0.01, min(self.max_position_size, position_size))
        
        logger.debug(
            f"Position size: {position_size:.2%} "
            f"(confidence={confidence_multiplier:.2f}, "
            f"vol={vol_multiplier:.2f}, "
            f"dd={dd_multiplier:.2f})"
        )
        
        return position_size
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        atr: Optional[float] = None
    ) -> float:
        """
        Calculate stop loss price.
        
        Args:
            entry_price: Entry price
            side: 'LONG' or 'SHORT'
            atr: Average True Range (optional, for dynamic stops)
        
        Returns:
            Stop loss price
        """
        # Use ATR if available for dynamic stops
        if atr:
            stop_distance = max(self.stop_loss_pct, atr * 1.5)
        else:
            stop_distance = self.stop_loss_pct
        
        if side == 'LONG':
            stop_loss = entry_price * (1 - stop_distance)
        else:  # SHORT
            stop_loss = entry_price * (1 + stop_distance)
        
        return stop_loss
    
    def calculate_take_profit(
        self,
        entry_price: float,
        side: str,
        risk_reward_ratio: float = 3.0
    ) -> float:
        """
        Calculate take profit price.
        
        Args:
            entry_price: Entry price
            side: 'LONG' or 'SHORT'
            risk_reward_ratio: Target risk/reward ratio
        
        Returns:
            Take profit price
        """
        # Use risk/reward ratio to set take profit
        profit_distance = self.stop_loss_pct * risk_reward_ratio
        profit_distance = max(self.take_profit_pct, profit_distance)
        
        if side == 'LONG':
            take_profit = entry_price * (1 + profit_distance)
        else:  # SHORT
            take_profit = entry_price * (1 - profit_distance)
        
        return take_profit
    
    # ========================================================================
    # Level 2: Daily-Level Risk Management
    # ========================================================================
    
    def check_daily_limits(
        self,
        current_equity: float,
        proposed_trade: Optional[Dict] = None
    ) -> RiskDecision:
        """
        Check daily trading limits.
        
        Args:
            current_equity: Current account equity
            proposed_trade: Optional proposed trade details
        
        Returns:
            RiskDecision with action and reason
        """
        # Reset daily counters if new day
        self._reset_daily_counters_if_needed()
        
        # Check daily loss limit
        daily_pnl = current_equity - self.daily_start_equity
        daily_pnl_pct = daily_pnl / self.daily_start_equity
        
        if daily_pnl_pct <= -self.max_daily_loss:
            log_risk_event(
                event_type="DAILY_LOSS_LIMIT_EXCEEDED",
                severity="CRITICAL",
                details=f"Daily loss: {daily_pnl_pct:.2%} (limit: {-self.max_daily_loss:.2%})",
                action="Halt all trading for the day"
            )
            
            self.trading_halted = True
            self.halt_reason = "Daily loss limit exceeded"
            
            return RiskDecision(
                action=ActionType.HALT_NEW_TRADES,
                level=RiskLevel.CRITICAL,
                reason=f"Daily loss limit exceeded: {daily_pnl_pct:.2%}",
                should_halt_trading=True
            )
        
        # Warning at 75% of limit
        if daily_pnl_pct <= -self.max_daily_loss * 0.75:
            log_risk_event(
                event_type="DAILY_LOSS_WARNING",
                severity="WARNING",
                details=f"Daily loss: {daily_pnl_pct:.2%} (75% of limit)",
                action="Reduce position sizes"
            )
            
            return RiskDecision(
                action=ActionType.REDUCE_SIZE,
                level=RiskLevel.WARNING,
                reason=f"Approaching daily loss limit: {daily_pnl_pct:.2%}",
                recommended_position_size=self.max_position_size * 0.5
            )
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            log_risk_event(
                event_type="DAILY_TRADE_LIMIT_EXCEEDED",
                severity="WARNING",
                details=f"Daily trades: {self.daily_trades} (limit: {self.max_daily_trades})",
                action="Halt new trades for the day"
            )
            
            return RiskDecision(
                action=ActionType.HALT_NEW_TRADES,
                level=RiskLevel.WARNING,
                reason=f"Daily trade limit reached: {self.daily_trades}",
                should_halt_trading=True
            )
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            log_risk_event(
                event_type="CONSECUTIVE_LOSS_LIMIT",
                severity="WARNING",
                details=f"Consecutive losses: {self.consecutive_losses}",
                action="Halt trading and review strategy"
            )
            
            self.trading_halted = True
            self.halt_reason = "Too many consecutive losses"
            
            return RiskDecision(
                action=ActionType.HALT_NEW_TRADES,
                level=RiskLevel.WARNING,
                reason=f"Too many consecutive losses: {self.consecutive_losses}",
                should_halt_trading=True
            )
        
        # All daily checks passed
        return RiskDecision(
            action=ActionType.ALLOW,
            level=RiskLevel.NORMAL,
            reason="All daily limits within acceptable range"
        )
    
    def _reset_daily_counters_if_needed(self):
        """Reset daily counters if new day."""
        current_date = datetime.now().date()
        
        if current_date > self.last_reset_date:
            logger.info(f"Resetting daily counters for new day: {current_date}")
            
            self.daily_trades = 0
            self.last_reset_date = current_date
            
            # Reset daily loss tracking (but not consecutive losses)
            if hasattr(self, 'equity_history') and self.equity_history:
                self.daily_start_equity = self.equity_history[-1]
            
            # Reset halt if it was due to daily limits
            if self.halt_reason in ["Daily loss limit exceeded", "Daily trade limit exceeded"]:
                self.trading_halted = False
                self.halt_reason = None
                logger.info("Trading resumed after daily reset")
    
    # ========================================================================
    # Level 3: System-Level Risk Management (Emergency Protection)
    # ========================================================================
    
    def check_system_limits(
        self,
        current_equity: float,
        current_volatility: float,
        avg_volatility: float,
        open_positions: int = 0
    ) -> RiskDecision:
        """
        Check system-level limits and detect emergency situations.
        
        Args:
            current_equity: Current account equity
            current_volatility: Current market volatility
            avg_volatility: Average historical volatility
            open_positions: Number of open positions
        
        Returns:
            RiskDecision with emergency actions if needed
        """
        # Check maximum drawdown (EMERGENCY SHUTDOWN)
        current_dd = self.calculate_current_drawdown(current_equity)
        
        if current_dd >= self.max_drawdown:
            log_risk_event(
                event_type="MAX_DRAWDOWN_EXCEEDED",
                severity="EMERGENCY",
                details=f"Drawdown: {current_dd:.2%} (limit: {self.max_drawdown:.2%})",
                action="EMERGENCY SHUTDOWN - Close all positions immediately"
            )
            
            self.emergency_shutdown = True
            
            return RiskDecision(
                action=ActionType.EMERGENCY_SHUTDOWN,
                level=RiskLevel.EMERGENCY,
                reason=f"Maximum drawdown exceeded: {current_dd:.2%}",
                should_close_positions=True,
                should_halt_trading=True
            )
        
        # Warning at 80% of max drawdown
        if current_dd >= self.max_drawdown * 0.8:
            log_risk_event(
                event_type="DRAWDOWN_WARNING",
                severity="CRITICAL",
                details=f"Drawdown: {current_dd:.2%} (80% of limit)",
                action="Close partial positions and reduce size"
            )
            
            return RiskDecision(
                action=ActionType.CLOSE_POSITIONS,
                level=RiskLevel.CRITICAL,
                reason=f"Approaching maximum drawdown: {current_dd:.2%}",
                recommended_position_size=self.max_position_size * 0.3
            )
        
        # Black Swan Detection (extreme volatility spike)
        if avg_volatility > 0:
            vol_ratio = current_volatility / avg_volatility
            
            if vol_ratio >= self.emergency_volatility_multiplier:
                log_risk_event(
                    event_type="BLACK_SWAN_DETECTED",
                    severity="EMERGENCY",
                    details=f"Volatility spike: {vol_ratio:.1f}x normal",
                    action="Emergency halt and close positions"
                )
                
                self.trading_halted = True
                self.halt_reason = "Black swan event detected"
                
                return RiskDecision(
                    action=ActionType.CLOSE_POSITIONS,
                    level=RiskLevel.EMERGENCY,
                    reason=f"Extreme volatility spike detected: {vol_ratio:.1f}x normal",
                    should_close_positions=True,
                    should_halt_trading=True
                )
        
        # Check Sharpe ratio (performance quality)
        if len(self.daily_returns) >= 30:  # Need at least 30 days
            sharpe = self.calculate_sharpe_ratio()
            
            if sharpe < self.min_sharpe_ratio:
                log_risk_event(
                    event_type="LOW_SHARPE_RATIO",
                    severity="WARNING",
                    details=f"Sharpe ratio: {sharpe:.2f} (minimum: {self.min_sharpe_ratio:.2f})",
                    action="Review strategy performance"
                )
                
                # Don't halt, but warn
                return RiskDecision(
                    action=ActionType.REDUCE_SIZE,
                    level=RiskLevel.WARNING,
                    reason=f"Low Sharpe ratio: {sharpe:.2f}",
                    recommended_position_size=self.max_position_size * 0.7
                )
        
        # All system checks passed
        return RiskDecision(
            action=ActionType.ALLOW,
            level=RiskLevel.NORMAL,
            reason="All system limits healthy"
        )
    
    # ========================================================================
    # Master Risk Check
    # ========================================================================
    
    def evaluate_trade(
        self,
        current_equity: float,
        signal_confidence: float,
        current_volatility: float,
        avg_volatility: float,
        open_positions: int = 0
    ) -> Tuple[bool, RiskDecision]:
        """
        Comprehensive risk evaluation for a proposed trade.
        
        Args:
            current_equity: Current account equity
            signal_confidence: Trading signal confidence
            current_volatility: Current volatility
            avg_volatility: Average volatility
            open_positions: Number of open positions
        
        Returns:
            Tuple of (can_trade, risk_decision)
        """
        # If emergency shutdown, reject immediately
        if self.emergency_shutdown:
            return False, RiskDecision(
                action=ActionType.EMERGENCY_SHUTDOWN,
                level=RiskLevel.EMERGENCY,
                reason="System in emergency shutdown state"
            )
        
        # If trading halted, reject
        if self.trading_halted:
            return False, RiskDecision(
                action=ActionType.HALT_NEW_TRADES,
                level=RiskLevel.WARNING,
                reason=f"Trading halted: {self.halt_reason}"
            )
        
        # Check system limits first (most critical)
        system_decision = self.check_system_limits(
            current_equity, current_volatility, avg_volatility, open_positions
        )
        
        if system_decision.action in [ActionType.EMERGENCY_SHUTDOWN, ActionType.CLOSE_POSITIONS]:
            return False, system_decision
        
        # Check daily limits
        daily_decision = self.check_daily_limits(current_equity)
        
        if daily_decision.action == ActionType.HALT_NEW_TRADES:
            return False, daily_decision
        
        # Calculate appropriate position size
        position_size = self.calculate_position_size(
            current_equity, signal_confidence, current_volatility, avg_volatility
        )
        
        # Apply any size reductions from risk decisions
        if system_decision.action == ActionType.REDUCE_SIZE:
            position_size = min(position_size, system_decision.recommended_position_size or position_size)
        
        if daily_decision.action == ActionType.REDUCE_SIZE:
            position_size = min(position_size, daily_decision.recommended_position_size or position_size)
        
        # Final decision
        can_trade = position_size >= 0.01  # Minimum viable position size
        
        final_decision = RiskDecision(
            action=ActionType.ALLOW if can_trade else ActionType.HALT_NEW_TRADES,
            level=max(system_decision.level, daily_decision.level, key=lambda x: list(RiskLevel).index(x)),
            reason="Trade approved" if can_trade else "Position size too small",
            recommended_position_size=position_size
        )
        
        return can_trade, final_decision
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def calculate_current_drawdown(self, current_equity: float) -> float:
        """Calculate current drawdown from peak."""
        self.peak_equity = max(self.peak_equity, current_equity)
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        return max(0, drawdown)
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from daily returns."""
        if len(self.daily_returns) < 2:
            return 0.0
        
        returns = np.array(self.daily_returns)
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return sharpe
    
    def update_equity(self, new_equity: float):
        """Update equity tracking."""
        if self.equity_history:
            daily_return = (new_equity - self.equity_history[-1]) / self.equity_history[-1]
            self.daily_returns.append(daily_return)
        
        self.equity_history.append(new_equity)
        
        # Keep only recent history (last 365 days)
        if len(self.equity_history) > 365:
            self.equity_history = self.equity_history[-365:]
            self.daily_returns = self.daily_returns[-365:]
    
    def record_trade_result(self, pnl: float):
        """Record trade result for consecutive loss tracking."""
        self.daily_trades += 1
        self.trade_results.append(pnl)
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Keep only recent trades
        if len(self.trade_results) > 100:
            self.trade_results = self.trade_results[-100:]
    
    def get_risk_status(self, current_equity: float) -> Dict:
        """Get current risk status summary."""
        current_dd = self.calculate_current_drawdown(current_equity)
        sharpe = self.calculate_sharpe_ratio() if len(self.daily_returns) >= 30 else None
        
        return {
            'emergency_shutdown': self.emergency_shutdown,
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'current_drawdown': current_dd,
            'drawdown_limit': self.max_drawdown,
            'daily_trades': self.daily_trades,
            'daily_trade_limit': self.max_daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_loss_limit': self.max_consecutive_losses,
            'sharpe_ratio': sharpe,
            'min_sharpe': self.min_sharpe_ratio,
        }


# Global risk manager instance
risk_manager = RiskManager()


if __name__ == "__main__":
    # Test risk management system
    print("=== Risk Management System Test ===\n")
    
    rm = RiskManager()
    
    # Test 1: Normal trading conditions
    print("Test 1: Normal conditions")
    can_trade, decision = rm.evaluate_trade(
        current_equity=10000,
        signal_confidence=0.8,
        current_volatility=0.02,
        avg_volatility=0.02,
        open_positions=1
    )
    print(f"Can trade: {can_trade}")
    print(f"Decision: {decision}\n")
    
    # Test 2: High drawdown
    print("Test 2: High drawdown (approaching limit)")
    rm.peak_equity = 10000
    can_trade, decision = rm.evaluate_trade(
        current_equity=9550,  # 4.5% drawdown (limit is 5%)
        signal_confidence=0.8,
        current_volatility=0.02,
        avg_volatility=0.02
    )
    print(f"Can trade: {can_trade}")
    print(f"Decision: {decision}\n")
    
    # Test 3: Volatility spike (Black Swan)
    print("Test 3: Extreme volatility spike")
    can_trade, decision = rm.evaluate_trade(
        current_equity=10000,
        signal_confidence=0.8,
        current_volatility=0.08,  # 4x normal
        avg_volatility=0.02
    )
    print(f"Can trade: {can_trade}")
    print(f"Decision: {decision}\n")
    
    # Test 4: Daily loss limit
    print("Test 4: Daily loss limit exceeded")
    rm.daily_start_equity = 10000
    can_trade, decision = rm.evaluate_trade(
        current_equity=9750,  # 2.5% daily loss (limit is 2%)
        signal_confidence=0.8,
        current_volatility=0.02,
        avg_volatility=0.02
    )
    print(f"Can trade: {can_trade}")
    print(f"Decision: {decision}\n")
    
    # Test 5: Risk status
    print("Test 5: Risk status summary")
    status = rm.get_risk_status(current_equity=9800)
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Risk management test completed!")
