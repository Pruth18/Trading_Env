"""
Base strategy class for the KITE trading system.
Provides the foundation for implementing trading strategies.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from kite.core.models import (
    Order, OrderType, OrderSide, OrderStatus, TimeInForce,
    Trade, Position, PositionSide, Asset, Portfolio, StrategyConfig
)
from kite.core.data import get_data_manager
from kite.utils.logging import get_strategy_logger


class Strategy(ABC):
    """
    Base class for all trading strategies.
    
    Provides common functionality for strategy implementation, including:
    - Data access
    - Order management
    - Position tracking
    - Performance metrics
    
    Subclasses must implement the `generate_signals` method to define the strategy logic.
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the strategy.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.name = config.name
        self.symbols = config.symbols
        self.parameters = config.parameters
        self.enabled = config.enabled
        
        # Initialize data manager
        self.data_manager = get_data_manager()
        
        # Initialize logger
        self.logger = get_strategy_logger(self.name)
        
        # Strategy state
        self.is_initialized = False
        self.is_running = False
        self.last_run_time = None
        
        # Trading data
        self.data: Dict[str, pd.DataFrame] = {}
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        
        # Performance tracking
        self.portfolio = Portfolio(name=f"{self.name} Portfolio")
        self.equity_curve = pd.DataFrame(columns=["timestamp", "equity", "cash", "holdings"])
        
        self.logger.info(f"Strategy '{self.name}' initialized with parameters: {self.parameters}")
    
    def initialize(self) -> bool:
        """
        Initialize the strategy before running.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Validate symbols
            if not self.symbols:
                self.logger.error("No symbols specified in strategy configuration")
                return False
            
            # Validate parameters
            self._validate_parameters()
            
            # Initialize data for each symbol
            for symbol in self.symbols:
                # Initialize empty position
                self.positions[symbol] = Position(symbol=symbol)
            
            # Set initialization flag
            self.is_initialized = True
            
            self.logger.info(f"Strategy '{self.name}' initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Strategy initialization failed: {str(e)}")
            return False
    
    def _validate_parameters(self) -> None:
        """
        Validate strategy parameters.
        
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # This is a placeholder for parameter validation
        # Subclasses should override this method to validate their specific parameters
        pass
    
    def load_data(self, start_date: datetime, end_date: datetime, interval: str = "1d") -> bool:
        """
        Load historical data for all symbols.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval (e.g., "1m", "5m", "1h", "1d")
            
        Returns:
            True if data loading was successful, False otherwise
        """
        try:
            for symbol in self.symbols:
                self.logger.info(f"Loading data for {symbol} from {start_date} to {end_date} ({interval})")
                df = self.data_manager.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval
                )
                
                if df.empty:
                    self.logger.warning(f"No data available for {symbol}")
                    continue
                
                self.data[symbol] = df
                self.logger.info(f"Loaded {len(df)} bars for {symbol}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            return False
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals based on the data.
        
        This method must be implemented by subclasses to define the strategy logic.
        
        Args:
            data: Dictionary of DataFrames with historical price data for each symbol
            
        Returns:
            Dictionary of signals for each symbol, where each signal is a dictionary
            with signal type, direction, strength, etc.
        """
        pass
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY
    ) -> Order:
        """
        Create a new order.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            order_type: Order type (MARKET/LIMIT/STOP/STOP_LIMIT)
            price: Limit price (required for LIMIT and STOP_LIMIT orders)
            stop_price: Stop price (required for STOP and STOP_LIMIT orders)
            time_in_force: Time in force (DAY/GTC/IOC/FOK)
            
        Returns:
            Created Order object
        """
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            status=OrderStatus.CREATED,
            strategy_id=self.name
        )
        
        self.orders[order.id] = order
        self.logger.info(f"Created order: {order.id}, {side.value} {quantity} {symbol} @ {price if price else 'MARKET'}")
        
        return order
    
    def execute_order(self, order: Order, price: Optional[float] = None) -> Trade:
        """
        Execute an order (for backtesting).
        
        In live trading, orders would be sent to the broker for execution.
        In backtesting, we simulate order execution here.
        
        Args:
            order: Order to execute
            price: Execution price (if None, use the latest price)
            
        Returns:
            Trade object representing the execution
        """
        # Get execution price
        if price is None:
            # Use the latest price from data
            if order.symbol in self.data:
                price = self.data[order.symbol]['close'].iloc[-1]
            else:
                raise ValueError(f"No data available for {order.symbol}")
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = price
        order.filled_at = datetime.now()
        order.updated_at = datetime.now()
        
        # Create trade
        trade = Trade(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=price,
            timestamp=datetime.now(),
            commission=price * order.quantity * 0.0020,  # 0.20% commission
            strategy_id=self.name
        )
        
        # Add trade to list
        self.trades.append(trade)
        
        # Update position
        self._update_position(trade)
        
        # Update portfolio
        self._update_portfolio(trade)
        
        self.logger.info(f"Executed order: {order.id}, {order.side.value} {order.quantity} {order.symbol} @ {price}")
        
        return trade
    
    def _update_position(self, trade: Trade) -> None:
        """
        Update position based on a trade.
        
        Args:
            trade: Trade to process
        """
        symbol = trade.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        position = self.positions[symbol]
        
        if trade.side == OrderSide.BUY:
            # Buying
            if position.side == PositionSide.FLAT:
                # Opening a new long position
                position.side = PositionSide.LONG
                position.quantity = trade.quantity
                position.average_price = trade.price
            elif position.side == PositionSide.LONG:
                # Adding to existing long position
                new_quantity = position.quantity + trade.quantity
                position.average_price = (position.average_price * position.quantity + trade.price * trade.quantity) / new_quantity
                position.quantity = new_quantity
            elif position.side == PositionSide.SHORT:
                # Covering a short position
                if trade.quantity < position.quantity:
                    # Partial cover
                    position.quantity -= trade.quantity
                    # Calculate realized PnL
                    position.realized_pnl += (position.average_price - trade.price) * trade.quantity
                elif trade.quantity == position.quantity:
                    # Full cover
                    position.realized_pnl += (position.average_price - trade.price) * trade.quantity
                    position.side = PositionSide.FLAT
                    position.quantity = 0
                    position.average_price = 0.0
                else:
                    # Cover and open long
                    position.realized_pnl += (position.average_price - trade.price) * position.quantity
                    remaining_qty = trade.quantity - position.quantity
                    position.side = PositionSide.LONG
                    position.quantity = remaining_qty
                    position.average_price = trade.price
        
        elif trade.side == OrderSide.SELL:
            # Selling
            if position.side == PositionSide.FLAT:
                # Opening a new short position
                position.side = PositionSide.SHORT
                position.quantity = trade.quantity
                position.average_price = trade.price
            elif position.side == PositionSide.SHORT:
                # Adding to existing short position
                new_quantity = position.quantity + trade.quantity
                position.average_price = (position.average_price * position.quantity + trade.price * trade.quantity) / new_quantity
                position.quantity = new_quantity
            elif position.side == PositionSide.LONG:
                # Closing a long position
                if trade.quantity < position.quantity:
                    # Partial close
                    position.quantity -= trade.quantity
                    # Calculate realized PnL
                    position.realized_pnl += (trade.price - position.average_price) * trade.quantity
                elif trade.quantity == position.quantity:
                    # Full close
                    position.realized_pnl += (trade.price - position.average_price) * trade.quantity
                    position.side = PositionSide.FLAT
                    position.quantity = 0
                    position.average_price = 0.0
                else:
                    # Close and open short
                    position.realized_pnl += (trade.price - position.average_price) * position.quantity
                    remaining_qty = trade.quantity - position.quantity
                    position.side = PositionSide.SHORT
                    position.quantity = remaining_qty
                    position.average_price = trade.price
        
        # Add trade to position
        if position.side != PositionSide.FLAT:
            position.open_trades.append(trade)
        else:
            position.closed_trades.append(trade)
        
        # Update unrealized PnL
        self._update_unrealized_pnl(symbol)
    
    def _update_unrealized_pnl(self, symbol: str) -> None:
        """
        Update unrealized PnL for a position.
        
        Args:
            symbol: Symbol to update
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        if position.side == PositionSide.FLAT:
            position.unrealized_pnl = 0.0
            return
        
        # Get current price
        if symbol in self.data:
            current_price = self.data[symbol]['close'].iloc[-1]
        else:
            return
        
        # Calculate unrealized PnL
        if position.side == PositionSide.LONG:
            position.unrealized_pnl = (current_price - position.average_price) * position.quantity
        elif position.side == PositionSide.SHORT:
            position.unrealized_pnl = (position.average_price - current_price) * position.quantity
    
    def _update_portfolio(self, trade: Trade) -> None:
        """
        Update portfolio based on a trade.
        
        Args:
            trade: Trade to process
        """
        # Update cash
        if trade.side == OrderSide.BUY:
            self.portfolio.cash -= trade.price * trade.quantity + trade.commission
        elif trade.side == OrderSide.SELL:
            self.portfolio.cash += trade.price * trade.quantity - trade.commission
        
        # Update positions
        self.portfolio.positions = self.positions
        
        # Update equity curve
        self._update_equity_curve()
    
    def _update_equity_curve(self) -> None:
        """
        Update the equity curve with the current portfolio value.
        """
        # Calculate total equity
        equity = self.portfolio.cash
        holdings = 0.0
        
        for symbol, position in self.positions.items():
            if position.side != PositionSide.FLAT:
                # Get current price
                if symbol in self.data:
                    current_price = self.data[symbol]['close'].iloc[-1]
                    position_value = position.quantity * current_price
                    holdings += position_value
                    equity += position_value
        
        # Add to equity curve
        self.equity_curve = self.equity_curve.append({
            "timestamp": datetime.now(),
            "equity": equity,
            "cash": self.portfolio.cash,
            "holdings": holdings
        }, ignore_index=True)
    
    def run(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Run the strategy on the provided data or the loaded data.
        
        Args:
            data: Dictionary of DataFrames with historical price data for each symbol
                 (if None, use the loaded data)
            
        Returns:
            Dictionary with strategy results
        """
        if not self.is_initialized:
            if not self.initialize():
                return {"success": False, "error": "Strategy initialization failed"}
        
        try:
            # Use provided data or loaded data
            if data is not None:
                self.data = data
            
            if not self.data:
                self.logger.error("No data available. Call load_data() first.")
                return {"success": False, "error": "No data available"}
            
            # Set running flag
            self.is_running = True
            self.last_run_time = datetime.now()
            
            # Generate signals
            signals = self.generate_signals(self.data)
            
            # Process signals
            for symbol, signal in signals.items():
                self._process_signal(symbol, signal)
            
            # Update all positions' unrealized PnL
            for symbol in self.symbols:
                self._update_unrealized_pnl(symbol)
            
            # Update equity curve
            self._update_equity_curve()
            
            # Set running flag
            self.is_running = False
            
            # Return results
            results = {
                "success": True,
                "signals": signals,
                "positions": {symbol: position.to_dict() for symbol, position in self.positions.items()},
                "portfolio": self.portfolio.to_dict(),
                "equity_curve": self.equity_curve.to_dict(orient="records"),
                "trades": [trade.to_dict() for trade in self.trades]
            }
            
            self.logger.info(f"Strategy '{self.name}' run completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Strategy run failed: {str(e)}")
            self.is_running = False
            return {"success": False, "error": str(e)}
    
    def _process_signal(self, symbol: str, signal: Dict[str, Any]) -> None:
        """
        Process a trading signal.
        
        Args:
            symbol: Trading symbol
            signal: Signal dictionary with signal type, direction, strength, etc.
        """
        # This is a placeholder for signal processing
        # Subclasses can override this method to implement custom signal processing logic
        
        # Example implementation:
        if "action" not in signal:
            return
        
        action = signal["action"]
        quantity = signal.get("quantity", 1)
        
        if action == "BUY":
            # Create and execute buy order
            order = self.create_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET
            )
            self.execute_order(order)
            
        elif action == "SELL":
            # Create and execute sell order
            order = self.create_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.MARKET
            )
            self.execute_order(order)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the strategy.
        
        Returns:
            Dictionary of performance metrics
        """
        if self.equity_curve.empty:
            return {}
        
        # Extract equity series
        equity = self.equity_curve["equity"]
        
        # Calculate returns
        returns = equity.pct_change().dropna()
        
        if len(returns) < 2:
            return {}
        
        # Calculate metrics
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        daily_returns = returns.resample("D").sum().dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        # Calculate drawdown
        drawdown = 1 - equity / equity.cummax()
        max_drawdown = drawdown.max()
        
        # Calculate win rate
        if self.trades:
            winning_trades = [t for t in self.trades if (t.side == OrderSide.BUY and t.price < self.data[t.symbol]['close'].iloc[-1]) or
                             (t.side == OrderSide.SELL and t.price > self.data[t.symbol]['close'].iloc[-1])]
            win_rate = len(winning_trades) / len(self.trades)
        else:
            win_rate = 0
        
        # Return metrics
        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": len(self.trades)
        }
        
        return metrics
    
    def reset(self) -> None:
        """
        Reset the strategy to its initial state.
        """
        # Reset trading data
        self.data = {}
        self.positions = {}
        self.orders = {}
        self.trades = []
        
        # Reset performance tracking
        self.portfolio = Portfolio(name=f"{self.name} Portfolio")
        self.equity_curve = pd.DataFrame(columns=["timestamp", "equity", "cash", "holdings"])
        
        # Reset state
        self.is_initialized = False
        self.is_running = False
        self.last_run_time = None
        
        self.logger.info(f"Strategy '{self.name}' reset to initial state")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert strategy to dictionary.
        
        Returns:
            Dictionary representation of the strategy
        """
        return {
            "name": self.name,
            "config": self.config.to_dict(),
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "positions": {symbol: position.to_dict() for symbol, position in self.positions.items()},
            "portfolio": self.portfolio.to_dict(),
            "metrics": self.calculate_metrics()
        }
