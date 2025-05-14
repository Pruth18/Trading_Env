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
        self.logger.info(f"Generating signals for RSI strategy with parameters: {self.parameters}")
        self.logger.info(f"Number of symbols in data: {len(data)}")
        signals = {}
        
        # Get parameters
        rsi_period = self.parameters["rsi_period"]
        oversold_level = self.parameters["oversold_level"]
        overbought_level = self.parameters["overbought_level"]
        position_size = self.parameters["position_size"]
        
        for symbol, df in data.items():
            self.logger.info(f"Processing symbol: {symbol} with {len(df)} data points")
            
            # Calculate RSI
            df = df.copy()
            df["rsi"] = self._calculate_rsi(df["close"], rsi_period)
            
            # Log RSI values for debugging
            self.logger.info(f"RSI values for {symbol} (last 5 bars): {df['rsi'].tail(5).tolist()}")
            self.logger.info(f"Min RSI: {df['rsi'].min():.2f}, Max RSI: {df['rsi'].max():.2f}")
            
            # Generate signals
            df["signal"] = 0
            df.loc[df["rsi"] < oversold_level, "signal"] = 1  # Buy signal
            df.loc[df["rsi"] > overbought_level, "signal"] = -1  # Sell signal
            
            # Get the latest data point
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None
            
            self.logger.info(f"Latest RSI for {symbol}: {latest['rsi']:.2f}")
            
            # Log oversold/overbought conditions
            self.logger.info(f"Oversold level: {oversold_level}, Overbought level: {overbought_level}")
            self.logger.info(f"Is {symbol} oversold? {latest['rsi'] < oversold_level}")
            self.logger.info(f"Is {symbol} overbought? {latest['rsi'] > overbought_level}")
            
            # Log current positions
            if symbol in self.positions:
                position = self.positions[symbol]
                self.logger.info(f"Current position for {symbol}: Side={position.side}, Quantity={position.quantity}")
            else:
                self.logger.info(f"No current position for {symbol}")
                
            # Log signal
            self.logger.info(f"Signal for {symbol}: {latest['signal']}")
            
            # Check if we have any buy/sell conditions
            buy_condition = latest["rsi"] < oversold_level
            sell_condition = latest["rsi"] > overbought_level
            mean_reversion_buy = prev is not None and latest["rsi"] > prev["rsi"] and prev["rsi"] < 40 and latest["rsi"] < 50
            mean_reversion_sell = prev is not None and latest["rsi"] < prev["rsi"] and prev["rsi"] > 60 and latest["rsi"] > 50
            
            self.logger.info(f"Buy condition: {buy_condition}, Sell condition: {sell_condition}")
            self.logger.info(f"Mean reversion buy: {mean_reversion_buy}, Mean reversion sell: {mean_reversion_sell}")
            
            # Determine action based on RSI
            action = None
            quantity = 0
            
            # Log position status before signal generation
            if symbol in self.positions:
                pos = self.positions[symbol]
                self.logger.info(f"Current position for {symbol}: Side={pos.side}, Quantity={pos.quantity}, Entry Price={pos.entry_price:.2f}")
            else:
                self.logger.info(f"No current position for {symbol}")
            
            # More aggressive signal generation
            # Buy when RSI is below oversold level
            if latest["rsi"] < oversold_level:
                # Check if we don't already have a position
                if symbol not in self.positions or self.positions[symbol].side == PositionSide.FLAT or self.positions[symbol].quantity == 0:
                    action = "BUY"
                    # Calculate position size based on portfolio value
                    price = latest["close"]
                    portfolio_value = self.portfolio.equity
                    quantity = int((portfolio_value * position_size) / price)
                    
                    self.logger.info(f"BUY signal for {symbol} at {price:.2f} (RSI: {latest['rsi']:.2f})")
                    self.logger.info(f"Order details: Action={action}, Quantity={quantity}, Portfolio Value={portfolio_value:.2f}")
                else:
                    self.logger.info(f"Already have a position in {symbol}, skipping BUY signal")
            
            # Sell when RSI is above overbought level
            elif latest["rsi"] > overbought_level:
                # Check if we have a position to sell
                if symbol in self.positions and self.positions[symbol].side == PositionSide.LONG and self.positions[symbol].quantity > 0:
                    action = "SELL"
                    quantity = self.positions[symbol].quantity
                    price = latest["close"]
                    position = self.positions[symbol]
                    profit_loss = (price - position.entry_price) * position.quantity
                    profit_loss_pct = (price / position.entry_price - 1) * 100
                    
                    self.logger.info(f"SELL signal for {symbol} at {price:.2f} (RSI: {latest['rsi']:.2f})")
                    self.logger.info(f"Order details: Action={action}, Quantity={quantity}, P&L=${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
                else:
                    self.logger.info(f"No position in {symbol} to sell, skipping SELL signal")
            
            # Add mean reversion logic
            # Buy when RSI starts to increase from a low level
            elif prev is not None and latest["rsi"] > prev["rsi"] and prev["rsi"] < 40 and latest["rsi"] < 50:
                if symbol not in self.positions or self.positions[symbol].side == PositionSide.FLAT or self.positions[symbol].quantity == 0:
                    action = "BUY"
                    price = latest["close"]
                    portfolio_value = self.portfolio.equity
                    quantity = int((portfolio_value * position_size * 0.5) / price)  # Half position size for mean reversion
                    
                    self.logger.info(f"BUY (mean reversion) signal for {symbol} at {price:.2f} (RSI: {latest['rsi']:.2f})")
                    self.logger.info(f"Order details: Action={action}, Quantity={quantity}, Portfolio Value={portfolio_value:.2f}")
                else:
                    self.logger.info(f"Already have a position in {symbol}, skipping mean reversion BUY signal")
            
            # Sell when RSI starts to decrease from a high level
            elif prev is not None and latest["rsi"] < prev["rsi"] and prev["rsi"] > 60 and latest["rsi"] > 50:
                if symbol in self.positions and self.positions[symbol].side == PositionSide.LONG and self.positions[symbol].quantity > 0:
                    action = "SELL"
                    quantity = self.positions[symbol].quantity
                    price = latest["close"]
                    position = self.positions[symbol]
                    profit_loss = (price - position.entry_price) * position.quantity
                    profit_loss_pct = (price / position.entry_price - 1) * 100
                    
                    self.logger.info(f"SELL (mean reversion) signal for {symbol} at {price:.2f} (RSI: {latest['rsi']:.2f})")
                    self.logger.info(f"Order details: Action={action}, Quantity={quantity}, P&L=${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
                else:
                    self.logger.info(f"No position in {symbol} to sell, skipping mean reversion SELL signal") 
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
