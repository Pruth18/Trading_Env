"""
RSI Strategy for the KITE trading system.
A strategy that generates buy/sell signals based on RSI (Relative Strength Index) values.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union

from kite.strategies.base import Strategy
from kite.core.models import (
    Order, OrderType, OrderSide, OrderStatus, TimeInForce,
    Trade, Position, PositionSide, StrategyConfig
)


class RSIStrategy(Strategy):
    """
    RSI (Relative Strength Index) Strategy.
    
    Generates buy signals when RSI falls below the oversold level and
    sell signals when RSI rises above the overbought level.
    
    Parameters:
    - rsi_period: Period for RSI calculation (default: 14)
    - oversold_level: RSI level for oversold condition (default: 30)
    - overbought_level: RSI level for overbought condition (default: 70)
    - position_size: Position size as a percentage of portfolio (default: 0.1)
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize the strategy."""
        super().__init__(config)
    
    def _validate_parameters(self) -> None:
        """Validate strategy parameters."""
        # Set default parameters if not provided
        if "rsi_period" not in self.parameters:
            self.parameters["rsi_period"] = 14
            self.logger.info("Using default rsi_period: 14")
        
        if "oversold_level" not in self.parameters:
            self.parameters["oversold_level"] = 30
            self.logger.info("Using default oversold_level: 30")
        
        if "overbought_level" not in self.parameters:
            self.parameters["overbought_level"] = 70
            self.logger.info("Using default overbought_level: 70")
        
        if "position_size" not in self.parameters:
            self.parameters["position_size"] = 0.1
            self.logger.info("Using default position_size: 0.1 (10% of portfolio)")
        
        # Validate parameter values
        if self.parameters["oversold_level"] >= self.parameters["overbought_level"]:
            raise ValueError("oversold_level must be less than overbought_level")
        
        if self.parameters["position_size"] <= 0 or self.parameters["position_size"] > 1:
            raise ValueError("position_size must be between 0 and 1")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index).
        
        Args:
            prices: Series of prices
            period: RSI period
            
        Returns:
            Series of RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals based on RSI values.
        
        Args:
            data: Dictionary of DataFrames with historical price data for each symbol
            
        Returns:
            Dictionary of signals for each symbol
        """
        signals = {}
        
        # Get parameters
        rsi_period = self.parameters["rsi_period"]
        oversold_level = self.parameters["oversold_level"]
        overbought_level = self.parameters["overbought_level"]
        position_size = self.parameters["position_size"]
        
        for symbol, df in data.items():
            # Calculate RSI
            df = df.copy()
            df["rsi"] = self._calculate_rsi(df["close"], period=rsi_period)
            
            # Generate signals
            df["signal"] = 0
            df.loc[df["rsi"] < oversold_level, "signal"] = 1  # Buy signal (oversold)
            df.loc[df["rsi"] > overbought_level, "signal"] = -1  # Sell signal (overbought)
            
            # Get the latest data point
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None
            
            # Determine action based on RSI
            action = None
            quantity = 0
            
            # Check if RSI just crossed below oversold level (buy signal)
            if prev is not None and latest["rsi"] < oversold_level and prev["rsi"] >= oversold_level:
                action = "BUY"
                # Calculate position size based on portfolio value
                price = latest["close"]
                portfolio_value = self.portfolio.equity
                quantity = int((portfolio_value * position_size) / price)
                
                self.logger.info(f"BUY signal for {symbol} at {price:.2f} (RSI: {latest['rsi']:.2f})")
            
            # Check if RSI just crossed above overbought level (sell signal)
            elif prev is not None and latest["rsi"] > overbought_level and prev["rsi"] <= overbought_level:
                action = "SELL"
                # Sell all holdings
                if symbol in self.positions:
                    position = self.positions[symbol]
                    if position.side == PositionSide.LONG and position.quantity > 0:
                        quantity = position.quantity
                        
                        self.logger.info(f"SELL signal for {symbol} at {latest['close']:.2f} (RSI: {latest['rsi']:.2f})")
            
            # Create signal dictionary
            signals[symbol] = {
                "action": action,
                "quantity": quantity,
                "price": latest["close"],
                "timestamp": latest.name if isinstance(latest.name, pd.Timestamp) else pd.Timestamp.now(),
                "indicators": {
                    "rsi": latest["rsi"],
                    "oversold_level": oversold_level,
                    "overbought_level": overbought_level
                }
            }
        
        return signals
