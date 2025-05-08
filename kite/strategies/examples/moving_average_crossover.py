"""
Moving Average Crossover Strategy for the KITE trading system.
A simple strategy that generates buy/sell signals based on moving average crossovers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union

from kite.strategies.base import Strategy
from kite.core.models import (
    Order, OrderType, OrderSide, OrderStatus, TimeInForce,
    Trade, Position, PositionSide, StrategyConfig
)


class MovingAverageCrossoverStrategy(Strategy):
    """
    Moving Average Crossover Strategy.
    
    Generates buy signals when the fast moving average crosses above the slow moving average,
    and sell signals when the fast moving average crosses below the slow moving average.
    
    Parameters:
    - fast_period: Period for the fast moving average (default: 20)
    - slow_period: Period for the slow moving average (default: 50)
    - signal_period: Period for the signal line (default: 9)
    - position_size: Position size as a percentage of portfolio (default: 0.1)
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize the strategy."""
        super().__init__(config)
    
    def _validate_parameters(self) -> None:
        """Validate strategy parameters."""
        # Set default parameters if not provided
        if "fast_period" not in self.parameters:
            self.parameters["fast_period"] = 20
            self.logger.info("Using default fast_period: 20")
        
        if "slow_period" not in self.parameters:
            self.parameters["slow_period"] = 50
            self.logger.info("Using default slow_period: 50")
        
        if "signal_period" not in self.parameters:
            self.parameters["signal_period"] = 9
            self.logger.info("Using default signal_period: 9")
        
        if "position_size" not in self.parameters:
            self.parameters["position_size"] = 0.1
            self.logger.info("Using default position_size: 0.1 (10% of portfolio)")
        
        # Validate parameter values
        if self.parameters["fast_period"] >= self.parameters["slow_period"]:
            raise ValueError("fast_period must be less than slow_period")
        
        if self.parameters["position_size"] <= 0 or self.parameters["position_size"] > 1:
            raise ValueError("position_size must be between 0 and 1")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data: Dictionary of DataFrames with historical price data for each symbol
            
        Returns:
            Dictionary of signals for each symbol
        """
        signals = {}
        
        # Get parameters
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]
        signal_period = self.parameters["signal_period"]
        position_size = self.parameters["position_size"]
        
        for symbol, df in data.items():
            # Calculate moving averages
            df = df.copy()
            df["fast_ma"] = df["close"].rolling(window=fast_period).mean()
            df["slow_ma"] = df["close"].rolling(window=slow_period).mean()
            
            # Calculate MACD
            df["macd"] = df["fast_ma"] - df["slow_ma"]
            df["signal_line"] = df["macd"].rolling(window=signal_period).mean()
            df["histogram"] = df["macd"] - df["signal_line"]
            
            # Generate signals
            df["signal"] = 0
            df.loc[df["macd"] > df["signal_line"], "signal"] = 1  # Buy signal
            df.loc[df["macd"] < df["signal_line"], "signal"] = -1  # Sell signal
            
            # Detect crossovers
            df["prev_signal"] = df["signal"].shift(1)
            df["crossover"] = df["signal"] != df["prev_signal"]
            
            # Get the latest data point
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None
            
            # Determine action based on crossover
            action = None
            quantity = 0
            
            if prev is not None and latest["crossover"]:
                if latest["signal"] == 1:  # Bullish crossover
                    action = "BUY"
                    # Calculate position size based on portfolio value
                    price = latest["close"]
                    portfolio_value = self.portfolio.equity
                    quantity = int((portfolio_value * position_size) / price)
                    
                    self.logger.info(f"BUY signal for {symbol} at {price:.2f}")
                
                elif latest["signal"] == -1:  # Bearish crossover
                    action = "SELL"
                    # Sell all holdings
                    if symbol in self.positions:
                        position = self.positions[symbol]
                        if position.side == PositionSide.LONG and position.quantity > 0:
                            quantity = position.quantity
                            
                            self.logger.info(f"SELL signal for {symbol} at {latest['close']:.2f}")
            
            # Create signal dictionary
            signals[symbol] = {
                "action": action,
                "quantity": quantity,
                "price": latest["close"],
                "timestamp": latest.name if isinstance(latest.name, pd.Timestamp) else pd.Timestamp.now(),
                "indicators": {
                    "fast_ma": latest["fast_ma"],
                    "slow_ma": latest["slow_ma"],
                    "macd": latest["macd"],
                    "signal_line": latest["signal_line"],
                    "histogram": latest["histogram"]
                }
            }
        
        return signals
