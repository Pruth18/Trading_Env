"""
Backtesting engine for the KITE trading system.
Provides functionality for backtesting trading strategies with historical data.
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Type, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger

from kite.core.models import (
    Order, OrderType, OrderSide, OrderStatus, TimeInForce,
    Trade, Position, PositionSide, Asset, Portfolio, 
    StrategyConfig, BacktestConfig, BacktestResult
)
from kite.core.data import get_data_manager
from kite.strategies.base import Strategy
from kite.utils.logging import get_backtest_logger, log_backtest_result
from kite.utils.config import Config, LOGS_DIR


class BacktestEngine:
    """
    Engine for backtesting trading strategies with historical data.
    
    Supports:
    - Single or multiple symbols
    - Multiple timeframes
    - Vectorized execution
    - Parameter optimization
    - Performance metrics calculation
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize the backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.id = config.id
        self.strategy_name = config.strategy_name
        self.start_date = config.start_date
        self.end_date = config.end_date
        self.initial_capital = config.initial_capital
        self.symbols = config.symbols
        self.interval = config.interval
        self.commission_rate = config.commission_rate
        self.slippage_rate = config.slippage_rate
        self.strategy_parameters = config.strategy_parameters
        
        # Initialize data manager
        self.data_manager = get_data_manager()
        
        # Initialize logger
        self.logger = get_backtest_logger(self.strategy_name, self.id)
        
        # Backtest state
        self.is_running = False
        self.start_time = None
        self.end_time = None
        self.data: Dict[str, pd.DataFrame] = {}
        
        self.logger.info(f"Backtest engine initialized for strategy '{self.strategy_name}' (ID: {self.id})")
        self.logger.info(f"Backtest period: {self.start_date} to {self.end_date}")
        self.logger.info(f"Symbols: {self.symbols}")
        self.logger.info(f"Interval: {self.interval}")
        self.logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        self.logger.info(f"Commission rate: {self.commission_rate:.4f}")
        self.logger.info(f"Slippage rate: {self.slippage_rate:.4f}")
        self.logger.info(f"Strategy parameters: {self.strategy_parameters}")
    
    def load_data(self) -> bool:
        """
        Load historical data for all symbols.
        
        Returns:
            True if data loading was successful, False otherwise
        """
        try:
            for symbol in self.symbols:
                self.logger.info(f"Loading data for {symbol} from {self.start_date} to {self.end_date} ({self.interval})")
                df = self.data_manager.get_historical_data(
                    symbol=symbol,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    interval=self.interval
                )
                
                if df.empty:
                    self.logger.warning(f"No data available for {symbol}")
                    continue
                
                self.data[symbol] = df
                self.logger.info(f"Loaded {len(df)} bars for {symbol}")
            
            if not self.data:
                self.logger.error("No data loaded for any symbol")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            return False
    
    def run(self, strategy_class: Type[Strategy]) -> Optional[BacktestResult]:
        """
        Run a backtest for a strategy.
        
        Args:
            strategy_class: Strategy class to backtest
            
        Returns:
            BacktestResult object if successful, None otherwise
        """
        if not self.data and not self.load_data():
            return None
        
        try:
            self.is_running = True
            self.start_time = datetime.now()
            
            self.logger.info(f"Starting backtest for strategy '{self.strategy_name}'")
            
            # Create strategy config
            strategy_config = StrategyConfig(
                name=self.strategy_name,
                symbols=self.symbols,
                parameters=self.strategy_parameters
            )
            
            # Initialize strategy
            strategy = strategy_class(strategy_config)
            
            # Initialize strategy with backtest settings
            strategy.portfolio.initial_capital = self.initial_capital
            strategy.portfolio.cash = self.initial_capital
            
            # Run strategy on historical data
            result = self._run_vectorized(strategy)
            
            self.is_running = False
            self.end_time = datetime.now()
            
            # Calculate execution time
            execution_time = (self.end_time - self.start_time).total_seconds()
            self.logger.info(f"Backtest completed in {execution_time:.2f} seconds")
            
            # Calculate performance metrics
            metrics = self._calculate_metrics(result)
            
            # Create backtest result
            backtest_result = BacktestResult(
                id=self.id,
                config=self.config,
                equity_curve=result["equity_curve"],
                trades=result["trades"],
                metrics=metrics,
                positions=result["positions"],
                logs=self._get_logs()
            )
            
            # Log backtest result
            log_backtest_result(self.strategy_name, self.id, metrics)
            
            return backtest_result
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            self.is_running = False
            self.end_time = datetime.now()
            return None
    
    def _run_vectorized(self, strategy: Strategy) -> Dict[str, Any]:
        """
        Run a vectorized backtest for a strategy.
        
        Args:
            strategy: Strategy instance to backtest
            
        Returns:
            Dictionary with backtest results
        """
        # Initialize strategy
        if not strategy.initialize():
            raise ValueError("Strategy initialization failed")
        
        # Set data
        strategy.data = self.data
        
        # Run strategy
        result = strategy.run()
        
        if not result["success"]:
            raise ValueError(f"Strategy run failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    def _calculate_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate performance metrics for a backtest result.
        
        Args:
            result: Dictionary with backtest results
            
        Returns:
            Dictionary of performance metrics
        """
        # Extract equity curve
        equity_curve = result["equity_curve"]
        if not equity_curve:
            return {}
        
        df = pd.DataFrame(equity_curve)
        
        # Calculate returns
        df["return"] = df["equity"].pct_change()
        
        # Calculate metrics
        total_return = (df["equity"].iloc[-1] / df["equity"].iloc[0]) - 1
        
        # Annualized return
        days = (df.index[-1] - df.index[0]).days if isinstance(df.index, pd.DatetimeIndex) else len(df)
        annualized_return = (1 + total_return) ** (252 / max(days, 1)) - 1
        
        # Daily returns
        daily_returns = df["return"].dropna()
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
        excess_returns = daily_returns - daily_risk_free
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Maximum drawdown
        df["cummax"] = df["equity"].cummax()
        df["drawdown"] = 1 - df["equity"] / df["cummax"]
        max_drawdown = df["drawdown"].max()
        
        # Win rate
        trades = result["trades"]
        if trades:
            winning_trades = [t for t in trades if 
                             (t["side"] == "BUY" and t["price"] < self.data[t["symbol"]]["close"].iloc[-1]) or
                             (t["side"] == "SELL" and t["price"] > self.data[t["symbol"]]["close"].iloc[-1])]
            win_rate = len(winning_trades) / len(trades)
        else:
            win_rate = 0
        
        # Profit factor
        gross_profit = sum(t["price"] * t["quantity"] for t in trades if t["side"] == "SELL")
        gross_loss = sum(t["price"] * t["quantity"] for t in trades if t["side"] == "BUY")
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        
        # Return metrics
        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "num_trades": len(trades)
        }
        
        return metrics
    
    def _get_logs(self) -> List[str]:
        """
        Get logs for the backtest.
        
        Returns:
            List of log entries
        """
        log_file = LOGS_DIR / f"kite_{self.strategy_name}_{self.id}_{datetime.now().strftime('%Y%m%d')}.log"
        
        if not log_file.exists():
            return []
        
        with open(log_file, "r") as f:
            logs = f.readlines()
        
        return logs


class ParameterOptimizer:
    """
    Optimizer for strategy parameters.
    
    Supports:
    - Grid search
    - Random search
    - Genetic algorithm (future)
    """
    
    def __init__(self, base_config: BacktestConfig, param_grid: Dict[str, List[Any]]):
        """
        Initialize the parameter optimizer.
        
        Args:
            base_config: Base backtest configuration
            param_grid: Grid of parameters to optimize
        """
        self.base_config = base_config
        self.param_grid = param_grid
        self.results: List[Tuple[Dict[str, Any], BacktestResult]] = []
        
        self.logger = logger.bind(optimizer=True)
        
        self.logger.info(f"Parameter optimizer initialized for strategy '{base_config.strategy_name}'")
        self.logger.info(f"Parameter grid: {param_grid}")
    
    def grid_search(self, strategy_class: Type[Strategy], max_workers: int = 1) -> List[Tuple[Dict[str, Any], BacktestResult]]:
        """
        Perform grid search parameter optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of (parameters, backtest_result) tuples, sorted by performance
        """
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations()
        
        self.logger.info(f"Running grid search with {len(param_combinations)} parameter combinations")
        
        # Run backtests
        if max_workers > 1:
            # Parallel execution
            self.results = self._run_parallel(strategy_class, param_combinations, max_workers)
        else:
            # Sequential execution
            self.results = self._run_sequential(strategy_class, param_combinations)
        
        # Sort results by performance (total return)
        self.results.sort(key=lambda x: x[1].metrics.get("total_return", float("-inf")), reverse=True)
        
        self.logger.info(f"Grid search completed with {len(self.results)} results")
        
        return self.results
    
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all combinations of parameters from the parameter grid.
        
        Returns:
            List of parameter dictionaries
        """
        import itertools
        
        # Get parameter names and values
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        param_combinations = []
        for combo in combinations:
            param_dict = {name: value for name, value in zip(param_names, combo)}
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def _run_sequential(self, strategy_class: Type[Strategy], param_combinations: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], BacktestResult]]:
        """
        Run backtests sequentially.
        
        Args:
            strategy_class: Strategy class to optimize
            param_combinations: List of parameter combinations to test
            
        Returns:
            List of (parameters, backtest_result) tuples
        """
        results = []
        
        for params in tqdm(param_combinations, desc="Optimizing parameters"):
            # Create backtest config with these parameters
            config = self._create_backtest_config(params)
            
            # Create backtest engine
            engine = BacktestEngine(config)
            
            # Run backtest
            result = engine.run(strategy_class)
            
            if result:
                results.append((params, result))
        
        return results
    
    def _run_parallel(self, strategy_class: Type[Strategy], param_combinations: List[Dict[str, Any]], max_workers: int) -> List[Tuple[Dict[str, Any], BacktestResult]]:
        """
        Run backtests in parallel.
        
        Args:
            strategy_class: Strategy class to optimize
            param_combinations: List of parameter combinations to test
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of (parameters, backtest_result) tuples
        """
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = []
            for params in param_combinations:
                future = executor.submit(self._run_backtest, strategy_class, params)
                futures.append((params, future))
            
            # Collect results as they complete
            for params, future in tqdm(as_completed(futures), total=len(futures), desc="Optimizing parameters"):
                result = future.result()
                if result:
                    results.append((params, result))
        
        return results
    
    def _run_backtest(self, strategy_class: Type[Strategy], params: Dict[str, Any]) -> Optional[BacktestResult]:
        """
        Run a single backtest with the given parameters.
        
        Args:
            strategy_class: Strategy class to optimize
            params: Parameters to test
            
        Returns:
            BacktestResult if successful, None otherwise
        """
        # Create backtest config with these parameters
        config = self._create_backtest_config(params)
        
        # Create backtest engine
        engine = BacktestEngine(config)
        
        # Run backtest
        return engine.run(strategy_class)
    
    def _create_backtest_config(self, params: Dict[str, Any]) -> BacktestConfig:
        """
        Create a backtest configuration with the given parameters.
        
        Args:
            params: Parameters to use
            
        Returns:
            BacktestConfig object
        """
        # Create a copy of the base config
        config = BacktestConfig(
            id=str(uuid.uuid4()),
            strategy_name=self.base_config.strategy_name,
            start_date=self.base_config.start_date,
            end_date=self.base_config.end_date,
            initial_capital=self.base_config.initial_capital,
            symbols=self.base_config.symbols,
            interval=self.base_config.interval,
            commission_rate=self.base_config.commission_rate,
            slippage_rate=self.base_config.slippage_rate,
            strategy_parameters=params
        )
        
        return config
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """
        Get the best parameters from the optimization results.
        
        Returns:
            Dictionary of best parameters
        """
        if not self.results:
            return {}
        
        # Return parameters of the best result
        return self.results[0][0]
    
    def get_best_result(self) -> Optional[BacktestResult]:
        """
        Get the best backtest result from the optimization results.
        
        Returns:
            Best BacktestResult object
        """
        if not self.results:
            return None
        
        # Return the best result
        return self.results[0][1]
