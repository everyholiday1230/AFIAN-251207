"""
Triple Barrier Labeling Method
================================

Revolutionary labeling method for creating action-based training labels.

Philosophy:
- Instead of predicting "will price go up?", we label "should I take action?"
- Each timestamp is labeled based on which barrier is hit first:
  * PROFIT_TARGET (+1.5%): Label as LONG
  * STOP_LOSS (-0.5%): Label as SHORT  
  * TIME_LIMIT (60 min): Label as NEUTRAL

This creates labels that directly correspond to optimal trading actions.

Reference: Marcos López de Prado - "Advances in Financial Machine Learning"
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from datetime import timedelta

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger("triple_barrier")


class TripleBarrierLabeler:
    """
    Create action-based labels using the Triple Barrier Method.
    
    Barriers:
    1. Upper Barrier (Profit Target): +1.5%
    2. Lower Barrier (Stop Loss): -0.5%
    3. Vertical Barrier (Time Limit): 60 minutes
    """
    
    def __init__(
        self,
        profit_target: float = None,
        stop_loss: float = None,
        time_limit_minutes: int = None
    ):
        """
        Initialize Triple Barrier Labeler.
        
        Args:
            profit_target: Profit target as decimal (e.g., 0.015 for 1.5%)
            stop_loss: Stop loss as decimal (e.g., 0.005 for 0.5%)
            time_limit_minutes: Maximum holding period in minutes
        """
        self.profit_target = profit_target or config.triple_barrier.profit_target
        self.stop_loss = stop_loss or config.triple_barrier.stop_loss_target
        self.time_limit_minutes = time_limit_minutes or config.triple_barrier.time_limit_minutes
        
        logger.info(
            f"Triple Barrier initialized: "
            f"Profit={self.profit_target:.2%}, "
            f"StopLoss={self.stop_loss:.2%}, "
            f"TimeLimit={self.time_limit_minutes}min"
        )
    
    def create_labels(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        timestamp_col: str = 'time'
    ) -> pd.DataFrame:
        """
        Create triple barrier labels for each timestamp.
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for prices
            timestamp_col: Column name for timestamps
        
        Returns:
            DataFrame with added label columns
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for labeling")
            return df
        
        # Ensure data is sorted
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        labels = []
        exit_prices = []
        exit_times = []
        returns = []
        holding_periods = []
        exit_reasons = []
        
        total = len(df)
        
        for i in range(total):
            if i % 1000 == 0:
                logger.debug(f"Labeling progress: {i}/{total} ({i/total*100:.1f}%)")
            
            entry_price = df.loc[i, price_col]
            entry_time = df.loc[i, timestamp_col]
            
            # Define barriers
            upper_barrier = entry_price * (1 + self.profit_target)
            lower_barrier = entry_price * (1 - self.stop_loss)
            time_barrier = entry_time + timedelta(minutes=self.time_limit_minutes)
            
            # Look forward to find which barrier is hit first
            label, exit_price, exit_time, return_pct, holding_min, exit_reason = self._find_first_barrier(
                df=df,
                start_idx=i + 1,
                entry_price=entry_price,
                entry_time=entry_time,
                upper_barrier=upper_barrier,
                lower_barrier=lower_barrier,
                time_barrier=time_barrier,
                price_col=price_col,
                timestamp_col=timestamp_col
            )
            
            labels.append(label)
            exit_prices.append(exit_price)
            exit_times.append(exit_time)
            returns.append(return_pct)
            holding_periods.append(holding_min)
            exit_reasons.append(exit_reason)
        
        # Add label columns
        df['tb_label'] = labels
        df['tb_exit_price'] = exit_prices
        df['tb_exit_time'] = exit_times
        df['tb_return'] = returns
        df['tb_holding_minutes'] = holding_periods
        df['tb_exit_reason'] = exit_reasons
        
        # Log label distribution
        label_counts = df['tb_label'].value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")
        
        return df
    
    def _find_first_barrier(
        self,
        df: pd.DataFrame,
        start_idx: int,
        entry_price: float,
        entry_time: pd.Timestamp,
        upper_barrier: float,
        lower_barrier: float,
        time_barrier: pd.Timestamp,
        price_col: str,
        timestamp_col: str
    ) -> Tuple[str, float, pd.Timestamp, float, int, str]:
        """
        Find which barrier is hit first.
        
        Returns:
            Tuple of (label, exit_price, exit_time, return_pct, holding_minutes, exit_reason)
        """
        # Check if we have enough future data
        if start_idx >= len(df):
            return 'NEUTRAL', entry_price, entry_time, 0.0, 0, 'INSUFFICIENT_DATA'
        
        # Iterate through future candles
        for idx in range(start_idx, len(df)):
            current_high = df.loc[idx, 'high'] if 'high' in df.columns else df.loc[idx, price_col]
            current_low = df.loc[idx, 'low'] if 'low' in df.columns else df.loc[idx, price_col]
            current_close = df.loc[idx, price_col]
            current_time = df.loc[idx, timestamp_col]
            
            # Check upper barrier (profit target)
            if current_high >= upper_barrier:
                exit_price = upper_barrier
                return_pct = (exit_price - entry_price) / entry_price
                holding_min = int((current_time - entry_time).total_seconds() / 60)
                
                return 'LONG', exit_price, current_time, return_pct, holding_min, 'PROFIT'
            
            # Check lower barrier (stop loss)
            if current_low <= lower_barrier:
                exit_price = lower_barrier
                return_pct = (exit_price - entry_price) / entry_price
                holding_min = int((current_time - entry_time).total_seconds() / 60)
                
                return 'SHORT', exit_price, current_time, return_pct, holding_min, 'LOSS'
            
            # Check time barrier
            if current_time >= time_barrier:
                exit_price = current_close
                return_pct = (exit_price - entry_price) / entry_price
                holding_min = int((current_time - entry_time).total_seconds() / 60)
                
                # If at time limit, classify based on current return
                if return_pct > 0.002:  # Small positive threshold (0.2%)
                    label = 'LONG'
                elif return_pct < -0.002:
                    label = 'SHORT'
                else:
                    label = 'NEUTRAL'
                
                return label, exit_price, current_time, return_pct, holding_min, 'TIMEOUT'
        
        # If we reached end of data without hitting any barrier
        exit_price = df.loc[len(df) - 1, price_col]
        exit_time = df.loc[len(df) - 1, timestamp_col]
        return_pct = (exit_price - entry_price) / entry_price
        holding_min = int((exit_time - entry_time).total_seconds() / 60)
        
        return 'NEUTRAL', exit_price, exit_time, return_pct, holding_min, 'END_OF_DATA'
    
    def create_balanced_labels(
        self,
        df: pd.DataFrame,
        balance_method: str = 'undersample'
    ) -> pd.DataFrame:
        """
        Create balanced dataset for training.
        
        Args:
            df: DataFrame with labels
            balance_method: 'undersample', 'oversample', or 'class_weight'
        
        Returns:
            Balanced DataFrame
        """
        if 'tb_label' not in df.columns:
            logger.error("DataFrame must have 'tb_label' column. Run create_labels() first.")
            return df
        
        # Remove rows with insufficient data
        df_clean = df[df['tb_exit_reason'] != 'INSUFFICIENT_DATA'].copy()
        df_clean = df_clean[df['tb_exit_reason'] != 'END_OF_DATA'].copy()
        
        label_counts = df_clean['tb_label'].value_counts()
        logger.info(f"Original label counts: {label_counts.to_dict()}")
        
        if balance_method == 'undersample':
            # Undersample to match smallest class
            min_count = label_counts.min()
            
            balanced_dfs = []
            for label in df_clean['tb_label'].unique():
                label_df = df_clean[df_clean['tb_label'] == label]
                sampled = label_df.sample(n=min(len(label_df), min_count), random_state=42)
                balanced_dfs.append(sampled)
            
            result = pd.concat(balanced_dfs, ignore_index=True)
            result = result.sample(frac=1, random_state=42).reset_index(drop=True)
            
            logger.info(f"Balanced dataset size: {len(result)}")
            logger.info(f"New label counts: {result['tb_label'].value_counts().to_dict()}")
            
            return result
        
        elif balance_method == 'oversample':
            # Oversample to match largest class
            max_count = label_counts.max()
            
            balanced_dfs = []
            for label in df_clean['tb_label'].unique():
                label_df = df_clean[df_clean['tb_label'] == label]
                if len(label_df) < max_count:
                    # Oversample with replacement
                    sampled = label_df.sample(n=max_count, replace=True, random_state=42)
                else:
                    sampled = label_df
                balanced_dfs.append(sampled)
            
            result = pd.concat(balanced_dfs, ignore_index=True)
            result = result.sample(frac=1, random_state=42).reset_index(drop=True)
            
            logger.info(f"Balanced dataset size: {len(result)}")
            logger.info(f"New label counts: {result['tb_label'].value_counts().to_dict()}")
            
            return result
        
        else:  # class_weight
            # Don't change data, return class weights for model training
            total = len(df_clean)
            class_weights = {}
            
            for label in df_clean['tb_label'].unique():
                count = len(df_clean[df_clean['tb_label'] == label])
                weight = total / (len(label_counts) * count)
                class_weights[label] = weight
            
            logger.info(f"Class weights: {class_weights}")
            df_clean.attrs['class_weights'] = class_weights
            
            return df_clean
    
    def get_label_statistics(self, df: pd.DataFrame) -> dict:
        """
        Get statistics about labels.
        
        Args:
            df: DataFrame with labels
        
        Returns:
            Dictionary with statistics
        """
        if 'tb_label' not in df.columns:
            return {}
        
        stats = {
            'total_samples': len(df),
            'label_counts': df['tb_label'].value_counts().to_dict(),
            'label_percentages': (df['tb_label'].value_counts(normalize=True) * 100).to_dict(),
        }
        
        # Statistics by exit reason
        if 'tb_exit_reason' in df.columns:
            stats['exit_reasons'] = df['tb_exit_reason'].value_counts().to_dict()
        
        # Return statistics by label
        if 'tb_return' in df.columns:
            for label in df['tb_label'].unique():
                label_data = df[df['tb_label'] == label]
                stats[f'{label}_avg_return'] = label_data['tb_return'].mean()
                stats[f'{label}_median_return'] = label_data['tb_return'].median()
        
        # Holding period statistics
        if 'tb_holding_minutes' in df.columns:
            stats['avg_holding_minutes'] = df['tb_holding_minutes'].mean()
            stats['median_holding_minutes'] = df['tb_holding_minutes'].median()
        
        return stats


# Global labeler instance
triple_barrier_labeler = TripleBarrierLabeler()


def quick_label(
    df: pd.DataFrame,
    profit_target: float = 0.015,
    stop_loss: float = 0.005,
    time_limit_minutes: int = 60
) -> pd.DataFrame:
    """
    Convenience function for quick labeling.
    
    Args:
        df: DataFrame with OHLCV data
        profit_target: Profit target percentage
        stop_loss: Stop loss percentage
        time_limit_minutes: Time limit in minutes
    
    Returns:
        Labeled DataFrame
    """
    labeler = TripleBarrierLabeler(
        profit_target=profit_target,
        stop_loss=stop_loss,
        time_limit_minutes=time_limit_minutes
    )
    
    return labeler.create_labels(df)


if __name__ == "__main__":
    # Test Triple Barrier Labeling
    print("=== Triple Barrier Labeling Test ===\n")
    
    # Create sample price data with trend
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=500, freq='1min')
    
    # Simulate price movement with drift
    returns = np.random.randn(500) * 0.001 + 0.0001  # Slight upward drift
    prices = 45000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'time': dates,
        'open': prices + np.random.randn(500) * 10,
        'high': prices + abs(np.random.randn(500) * 50),
        'low': prices - abs(np.random.randn(500) * 50),
        'close': prices,
        'volume': np.random.rand(500) * 100,
    })
    
    print("Sample Data:")
    print(df[['time', 'close']].head(10))
    print()
    
    # Create labels
    print("Creating Triple Barrier labels...")
    labeler = TripleBarrierLabeler(
        profit_target=0.015,  # 1.5%
        stop_loss=0.005,      # 0.5%
        time_limit_minutes=60
    )
    
    labeled_df = labeler.create_labels(df)
    
    print("\nLabeled Data:")
    print(labeled_df[['time', 'close', 'tb_label', 'tb_return', 'tb_exit_reason']].head(20))
    print()
    
    # Get statistics
    stats = labeler.get_label_statistics(labeled_df)
    
    print("Label Statistics:")
    print(f"Total Samples: {stats['total_samples']}")
    print(f"\nLabel Distribution:")
    for label, count in stats['label_counts'].items():
        pct = stats['label_percentages'][label]
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print(f"\nExit Reasons:")
    for reason, count in stats['exit_reasons'].items():
        print(f"  {reason}: {count}")
    
    print(f"\nAverage Returns by Label:")
    for label in ['LONG', 'SHORT', 'NEUTRAL']:
        if f'{label}_avg_return' in stats:
            avg_return = stats[f'{label}_avg_return']
            print(f"  {label}: {avg_return:.4%}")
    
    print(f"\nHolding Period:")
    print(f"  Average: {stats['avg_holding_minutes']:.1f} minutes")
    print(f"  Median: {stats['median_holding_minutes']:.1f} minutes")
    
    # Test balanced labels
    print("\n\nCreating Balanced Dataset...")
    balanced_df = labeler.create_balanced_labels(labeled_df, balance_method='undersample')
    
    print("\nBalanced Label Distribution:")
    print(balanced_df['tb_label'].value_counts())
    
    print("\n✅ Triple Barrier labeling test completed!")
