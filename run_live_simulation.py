#!/usr/bin/env python
"""
Live Trading Simulation for KITE Trading System

This script simulates live trading by running strategies on historical data
and then simulating real-time updates. It allows for continuous running and
monitoring of strategy performance.
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Any, Optional, Union
import threading
import signal
import sys

from loguru import logger

from kite.backtest.engine import BacktestEngine
from kite.backtest.config import BacktestConfig
from kite.core.models import StrategyConfig, Order, Trade, Position
from kite.strategies.examples.rsi_strategy import RSIStrategy
from kite.strategies.examples.moving_average_crossover import MovingAverageCrossoverStrategy
from kite.analysis.plots import plot_equity_curve
from kite.utils.logging import setup_logger
from kite.core.data import get_data_manager


class LiveSimulation:
    """Live trading simulation class."""
    
    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        interval: str = "1d",
        initial_capital: float = 100000.0,
        commission_rate: float = 0.002,
        slippage_rate: float = 0.0005,
        output_dir: str = "simulation_results",
        history_days: int = 180,
        update_interval_seconds: int = 60
    ):
        """
        Initialize the live simulation.
        
        Args:
            strategy_name: Name of the strategy to run
            symbols: List of symbols to trade
            interval: Data interval
            initial_capital: Initial capital
            commission_rate: Commission rate as a decimal
            slippage_rate: Slippage rate as a decimal
            output_dir: Directory to save results
            history_days: Number of days of historical data to load
            update_interval_seconds: Interval between updates in seconds
        """
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.interval = interval
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.history_days = history_days
        self.update_interval_seconds = update_interval_seconds
        
        # Create strategy configuration
        if strategy_name.lower() == "rsi":
            self.strategy_config = self._create_rsi_strategy_config()
        elif strategy_name.lower() == "ma_crossover":
            self.strategy_config = self._create_ma_crossover_strategy_config()
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Initialize data manager
        self.data_manager = get_data_manager()
        
        # Initialize simulation state
        self.start_time = datetime.now()
        self.current_time = self.start_time
        self.is_running = False
        self.trades = []
        self.equity_curve = pd.DataFrame(columns=["timestamp", "equity", "returns", "drawdown"])
        self.equity_curve.set_index("timestamp", inplace=True)
        
        # Initialize metrics
        self.metrics = {
            "total_return": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0
        }
        
        # Create simulation ID
        self.simulation_id = f"{strategy_name.lower().replace(' ', '_')}_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize strategy
        self._initialize_strategy()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _create_rsi_strategy_config(self) -> StrategyConfig:
        """Create RSI strategy configuration."""
        return StrategyConfig(
            name="RSI Strategy",
            symbols=self.symbols,
            parameters={
                "rsi_period": 8,
                "oversold_level": 40,
                "overbought_level": 60,
                "position_size": 0.2
            }
        )
    
    def _create_ma_crossover_strategy_config(self) -> StrategyConfig:
        """Create Moving Average Crossover strategy configuration."""
        return StrategyConfig(
            name="MA Crossover",
            symbols=self.symbols,
            parameters={
                "fast_period": 10,
                "slow_period": 30,
                "signal_period": 5,
                "position_size": 0.15
            }
        )
    
    def _initialize_strategy(self) -> None:
        """Initialize the strategy with historical data."""
        logger.info(f"Initializing {self.strategy_name} strategy with {self.history_days} days of historical data")
        
        # Create backtest configuration for historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.history_days)
        
        backtest_config = BacktestConfig(
            id=f"{self.simulation_id}_init",
            start_date=start_date,
            end_date=end_date,
            interval=self.interval,
            initial_capital=self.initial_capital,
            commission_rate=self.commission_rate,
            slippage_rate=self.slippage_rate
        )
        
        # Initialize backtest engine
        self.engine = BacktestEngine(self.strategy_config, backtest_config)
        
        # Load historical data
        self.engine.load_data()
        
        # Initialize strategy
        if self.strategy_name.lower() == "rsi":
            self.strategy = RSIStrategy(self.strategy_config)
        elif self.strategy_name.lower() == "ma_crossover":
            self.strategy = MovingAverageCrossoverStrategy(self.strategy_config)
        
        self.strategy.initialize(self.engine.data, self.initial_capital)
        
        # Run strategy on historical data
        logger.info("Running strategy on historical data...")
        self.strategy.run(self.engine.data)
        
        # Store initial state
        self.portfolio = self.strategy.portfolio
        self.positions = self.strategy.positions
        self.trades = self.strategy.trades
        self.equity_curve = self.strategy.equity_curve
        
        logger.info(f"Strategy initialized with {len(self.trades)} historical trades")
        logger.info(f"Current equity: ${self.portfolio.equity:.2f}")
        logger.info(f"Current positions: {len(self.positions)}")
        
        # Save initial state
        self._save_state()
    
    def _generate_new_bar(self) -> Dict[str, pd.DataFrame]:
        """
        Generate a new price bar for each symbol.
        In a real implementation, this would fetch the latest data from the API.
        """
        new_data = {}
        
        for symbol in self.symbols:
            # Get the last known data for this symbol
            last_data = self.engine.data[symbol].iloc[-1].copy()
            
            # Generate a new bar with some random price movement
            new_bar = pd.DataFrame(index=[self.current_time])
            
            # Random price movement between -1% and 1%
            price_change = np.random.normal(0, 0.01)
            close_price = last_data["close"] * (1 + price_change)
            
            # Generate OHLC data
            open_price = last_data["close"]  # Open at previous close
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.003)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.003)))
            
            # Ensure high >= open, close and low <= open, close
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Random volume
            volume = int(last_data["volume"] * (1 + np.random.normal(0, 0.1)))
            volume = max(volume, 100)  # Ensure positive volume
            
            # Create new bar
            new_bar["open"] = open_price
            new_bar["high"] = high_price
            new_bar["low"] = low_price
            new_bar["close"] = close_price
            new_bar["volume"] = volume
            new_bar["symbol"] = symbol
            
            # Combine with existing data
            new_data[symbol] = pd.concat([self.engine.data[symbol], new_bar])
        
        return new_data
    
    def _update(self) -> None:
        """Update the simulation with new data."""
        # Update current time
        self.current_time = datetime.now()
        
        # Generate new data
        new_data = self._generate_new_bar()
        
        # Run strategy on new data
        signals = self.strategy.generate_signals(new_data)
        
        # Process signals
        for symbol, signal in signals.items():
            if signal["action"] == "BUY":
                # Create and execute buy order
                order = Order(
                    symbol=symbol,
                    quantity=signal["quantity"],
                    side="BUY",
                    order_type="MARKET",
                    limit_price=None,
                    status="NEW"
                )
                
                self.strategy.create_order(order)
                self.strategy.execute_order(order, signal["price"])
                
                logger.info(f"BUY {signal['quantity']} {symbol} at ${signal['price']:.2f}")
            
            elif signal["action"] == "SELL":
                # Create and execute sell order
                order = Order(
                    symbol=symbol,
                    quantity=signal["quantity"],
                    side="SELL",
                    order_type="MARKET",
                    limit_price=None,
                    status="NEW"
                )
                
                self.strategy.create_order(order)
                self.strategy.execute_order(order, signal["price"])
                
                logger.info(f"SELL {signal['quantity']} {symbol} at ${signal['price']:.2f}")
        
        # Update portfolio and positions
        self.portfolio = self.strategy.portfolio
        self.positions = self.strategy.positions
        self.trades = self.strategy.trades
        self.equity_curve = self.strategy.equity_curve
        
        # Update metrics
        self._update_metrics()
        
        # Save state
        self._save_state()
        
        # Log current state
        logger.info(f"Update at {self.current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Current equity: ${self.portfolio.equity:.2f}")
        logger.info(f"Total return: {self.metrics['total_return']:.2f}%")
        logger.info(f"Number of trades: {self.metrics['num_trades']}")
        logger.info(f"Win rate: {self.metrics['win_rate']:.2f}%")
        logger.info(f"Max drawdown: {self.metrics['max_drawdown']:.2f}%")
        
        # Log current positions
        if self.positions:
            logger.info("Current positions:")
            for symbol, position in self.positions.items():
                if position.quantity > 0:
                    logger.info(f"  {symbol}: {position.quantity} shares at ${position.entry_price:.2f}")
    
    def _update_metrics(self) -> None:
        """Update performance metrics."""
        if self.equity_curve.empty:
            return
        
        # Calculate total return
        initial_equity = self.initial_capital
        current_equity = self.portfolio.equity
        total_return = (current_equity / initial_equity - 1) * 100
        self.metrics["total_return"] = total_return
        
        # Calculate number of trades
        self.metrics["num_trades"] = len(self.trades)
        
        # Calculate win rate
        if self.trades:
            winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
            self.metrics["win_rate"] = (winning_trades / len(self.trades)) * 100
        
        # Calculate profit factor
        gross_profit = sum(trade.pnl for trade in self.trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in self.trades if trade.pnl < 0))
        self.metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate max drawdown
        if "drawdown" in self.equity_curve.columns:
            self.metrics["max_drawdown"] = self.equity_curve["drawdown"].max() * 100
    
    def _save_state(self) -> None:
        """Save the current state of the simulation."""
        # Save equity curve
        equity_file = self.output_dir / f"{self.simulation_id}_equity.csv"
        self.equity_curve.to_csv(equity_file)
        
        # Save trades
        trades_file = self.output_dir / f"{self.simulation_id}_trades.csv"
        trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
        if not trades_df.empty:
            trades_df.to_csv(trades_file, index=False)
        
        # Save metrics
        metrics_file = self.output_dir / f"{self.simulation_id}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=4)
        
        # Create plots
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot equity curve
        equity_plot_file = plots_dir / f"{self.simulation_id}_equity.png"
        plot_equity_curve(self.equity_curve, self.strategy_config.name, equity_plot_file)
    
    def _signal_handler(self, sig, frame) -> None:
        """Handle signals for graceful shutdown."""
        logger.info("Received shutdown signal, stopping simulation...")
        self.stop()
        sys.exit(0)
    
    def start(self) -> None:
        """Start the simulation."""
        if self.is_running:
            logger.warning("Simulation is already running")
            return
        
        self.is_running = True
        logger.info(f"Starting {self.strategy_name} simulation at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            while self.is_running:
                # Update simulation
                self._update()
                
                # Sleep until next update
                time.sleep(self.update_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
            self.stop()
        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
            self.stop()
            raise
    
    def stop(self) -> None:
        """Stop the simulation."""
        if not self.is_running:
            logger.warning("Simulation is not running")
            return
        
        self.is_running = False
        logger.info(f"Stopping simulation after {(datetime.now() - self.start_time).total_seconds() / 60:.1f} minutes")
        
        # Final update and save
        self._update()
        
        # Log final results
        logger.success("Simulation completed!")
        logger.info(f"Final equity: ${self.portfolio.equity:.2f}")
        logger.info(f"Total return: {self.metrics['total_return']:.2f}%")
        logger.info(f"Number of trades: {self.metrics['num_trades']}")
        logger.info(f"Win rate: {self.metrics['win_rate']:.2f}%")
        logger.info(f"Max drawdown: {self.metrics['max_drawdown']:.2f}%")
        
        # Log results location
        logger.info(f"Results saved to {self.output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run live trading simulation for KITE trading system")
    
    parser.add_argument("--strategy", type=str, default="rsi", choices=["rsi", "ma_crossover"],
                        help="Strategy to simulate (default: rsi)")
    
    parser.add_argument("--symbols", type=str, default="RELIANCE,TCS,INFY",
                        help="Comma-separated list of symbols to trade (default: RELIANCE,TCS,INFY)")
    
    parser.add_argument("--interval", type=str, default="1d", choices=["1m", "5m", "15m", "30m", "1h", "1d", "1w"],
                        help="Data interval (default: 1d)")
    
    parser.add_argument("--initial-capital", type=float, default=100000.0,
                        help="Initial capital for simulation (default: 100000.0)")
    
    parser.add_argument("--commission-rate", type=float, default=0.002,
                        help="Commission rate as a decimal (default: 0.002 = 0.2%)")
    
    parser.add_argument("--slippage-rate", type=float, default=0.0005,
                        help="Slippage rate as a decimal (default: 0.0005 = 0.05%)")
    
    parser.add_argument("--output-dir", type=str, default="simulation_results",
                        help="Directory to save simulation results (default: simulation_results)")
    
    parser.add_argument("--history-days", type=int, default=180,
                        help="Number of days of historical data to load (default: 180)")
    
    parser.add_argument("--update-interval", type=int, default=60,
                        help="Interval between updates in seconds (default: 60)")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logger
    setup_logger()
    
    # Parse symbols
    symbols = args.symbols.split(",")
    
    # Create and start simulation
    simulation = LiveSimulation(
        strategy_name=args.strategy,
        symbols=symbols,
        interval=args.interval,
        initial_capital=args.initial_capital,
        commission_rate=args.commission_rate,
        slippage_rate=args.slippage_rate,
        output_dir=args.output_dir,
        history_days=args.history_days,
        update_interval_seconds=args.update_interval
    )
    
    # Start simulation
    simulation.start()


if __name__ == "__main__":
    main()
