#!/usr/bin/env python
"""
Continuous Backtest Runner for KITE Trading System

This script allows running backtests over extended periods with detailed performance tracking.
It supports running multiple strategies in parallel and saving results for later analysis.
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Any, Optional, Union

from loguru import logger

from kite.backtest.engine import BacktestEngine
from kite.backtest.config import BacktestConfig
from kite.core.models import StrategyConfig
from kite.strategies.examples.rsi_strategy import RSIStrategy
from kite.strategies.examples.moving_average_crossover import MovingAverageCrossoverStrategy
from kite.analysis.plots import plot_equity_curve, plot_drawdown, plot_returns_distribution
from kite.utils.logging import setup_logger, log_backtest_result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run continuous backtest for KITE trading system")
    
    parser.add_argument("--strategy", type=str, default="rsi", choices=["rsi", "ma_crossover", "all"],
                        help="Strategy to backtest (default: rsi)")
    
    parser.add_argument("--symbols", type=str, default="RELIANCE,TCS,INFY",
                        help="Comma-separated list of symbols to trade (default: RELIANCE,TCS,INFY)")
    
    parser.add_argument("--start-date", type=str, default="",
                        help="Start date for backtest in YYYY-MM-DD format (default: 1 year ago)")
    
    parser.add_argument("--end-date", type=str, default="",
                        help="End date for backtest in YYYY-MM-DD format (default: today)")
    
    parser.add_argument("--interval", type=str, default="1d", choices=["1m", "5m", "15m", "30m", "1h", "1d", "1w"],
                        help="Data interval (default: 1d)")
    
    parser.add_argument("--initial-capital", type=float, default=100000.0,
                        help="Initial capital for backtest (default: 100000.0)")
    
    parser.add_argument("--commission-rate", type=float, default=0.002,
                        help="Commission rate as a decimal (default: 0.002 = 0.2%)")
    
    parser.add_argument("--slippage-rate", type=float, default=0.0005,
                        help="Slippage rate as a decimal (default: 0.0005 = 0.05%)")
    
    parser.add_argument("--output-dir", type=str, default="backtest_results",
                        help="Directory to save backtest results (default: backtest_results)")
    
    return parser.parse_args()


def create_rsi_strategy_config(symbols: List[str]) -> StrategyConfig:
    """Create RSI strategy configuration."""
    return StrategyConfig(
        name="RSI Strategy",
        symbols=symbols,
        parameters={
            "rsi_period": 8,  # Shorter RSI period for more signals
            "oversold_level": 40,  # Higher oversold level to ensure buy signals
            "overbought_level": 60,  # Lower overbought level to ensure sell signals
            "position_size": 0.2  # Position size as percentage of portfolio
        }
    )


def create_ma_crossover_strategy_config(symbols: List[str]) -> StrategyConfig:
    """Create Moving Average Crossover strategy configuration."""
    return StrategyConfig(
        name="MA Crossover",
        symbols=symbols,
        parameters={
            "fast_period": 10,  # Shorter fast period for more signals
            "slow_period": 30,  # Shorter slow period for more signals
            "signal_period": 5,  # Shorter signal period
            "position_size": 0.15  # Position size as percentage of portfolio
        }
    )


def run_backtest(
    strategy_config: StrategyConfig,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1d",
    initial_capital: float = 100000.0,
    commission_rate: float = 0.002,
    slippage_rate: float = 0.0005,
    output_dir: str = "backtest_results"
) -> Dict[str, Any]:
    """
    Run a backtest for the specified strategy and parameters.
    
    Args:
        strategy_config: Strategy configuration
        start_date: Start date for backtest
        end_date: End date for backtest
        interval: Data interval
        initial_capital: Initial capital for backtest
        commission_rate: Commission rate as a decimal
        slippage_rate: Slippage rate as a decimal
        output_dir: Directory to save backtest results
        
    Returns:
        Dictionary with backtest results
    """
    # Create a unique ID for this backtest
    backtest_id = f"{strategy_config.name.lower().replace(' ', '_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    
    # Create backtest configuration
    backtest_config = BacktestConfig(
        id=backtest_id,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate
    )
    
    # Initialize backtest engine
    engine = BacktestEngine(strategy_config, backtest_config)
    
    # Run backtest
    logger.info(f"Running backtest for {strategy_config.name} from {start_date} to {end_date}")
    logger.info(f"Symbols: {strategy_config.symbols}")
    logger.info(f"Parameters: {strategy_config.parameters}")
    
    result = engine.run()
    
    # Log backtest result
    log_backtest_result(strategy_config.name, backtest_id, result.metrics)
    
    # Save backtest results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save metrics to JSON
    metrics_file = output_path / f"{backtest_id}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(
            {k: float(v) if isinstance(v, np.float64) else v for k, v in result.metrics.items()},
            f,
            indent=4
        )
    
    # Save equity curve to CSV
    equity_file = output_path / f"{backtest_id}_equity.csv"
    result.equity_curve.to_csv(equity_file)
    
    # Save trades to CSV
    trades_file = output_path / f"{backtest_id}_trades.csv"
    trades_df = pd.DataFrame([t.to_dict() for t in result.trades])
    if not trades_df.empty:
        trades_df.to_csv(trades_file, index=False)
    
    # Create plots
    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot equity curve
    equity_plot_file = plots_dir / f"{backtest_id}_equity.png"
    plot_equity_curve(result.equity_curve, strategy_config.name, equity_plot_file)
    
    # Plot drawdown
    drawdown_plot_file = plots_dir / f"{backtest_id}_drawdown.png"
    plot_drawdown(result.equity_curve, strategy_config.name, drawdown_plot_file)
    
    # Plot returns distribution
    returns_plot_file = plots_dir / f"{backtest_id}_returns.png"
    plot_returns_distribution(result.equity_curve, strategy_config.name, returns_plot_file)
    
    logger.success(f"Backtest completed for {strategy_config.name}")
    logger.info(f"Results saved to {output_path}")
    
    # Print summary metrics
    logger.info("Performance Metrics:")
    for key, value in result.metrics.items():
        if isinstance(value, (int, float, np.float64)):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return {
        "id": backtest_id,
        "strategy": strategy_config.name,
        "metrics": result.metrics,
        "equity_curve": result.equity_curve,
        "trades": result.trades
    }


def run_continuous_backtest(
    strategy_name: str,
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    interval: str = "1d",
    initial_capital: float = 100000.0,
    commission_rate: float = 0.002,
    slippage_rate: float = 0.0005,
    output_dir: str = "backtest_results"
) -> Dict[str, Any]:
    """
    Run a continuous backtest for the specified strategy.
    
    Args:
        strategy_name: Name of the strategy to run
        symbols: List of symbols to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        interval: Data interval
        initial_capital: Initial capital for backtest
        commission_rate: Commission rate as a decimal
        slippage_rate: Slippage rate as a decimal
        output_dir: Directory to save backtest results
        
    Returns:
        Dictionary with backtest results
    """
    if strategy_name.lower() == "rsi":
        strategy_config = create_rsi_strategy_config(symbols)
        return run_backtest(
            strategy_config,
            start_date,
            end_date,
            interval,
            initial_capital,
            commission_rate,
            slippage_rate,
            output_dir
        )
    elif strategy_name.lower() == "ma_crossover":
        strategy_config = create_ma_crossover_strategy_config(symbols)
        return run_backtest(
            strategy_config,
            start_date,
            end_date,
            interval,
            initial_capital,
            commission_rate,
            slippage_rate,
            output_dir
        )
    elif strategy_name.lower() == "all":
        results = {}
        
        # Run RSI strategy
        rsi_config = create_rsi_strategy_config(symbols)
        results["rsi"] = run_backtest(
            rsi_config,
            start_date,
            end_date,
            interval,
            initial_capital,
            commission_rate,
            slippage_rate,
            output_dir
        )
        
        # Run MA Crossover strategy
        ma_config = create_ma_crossover_strategy_config(symbols)
        results["ma_crossover"] = run_backtest(
            ma_config,
            start_date,
            end_date,
            interval,
            initial_capital,
            commission_rate,
            slippage_rate,
            output_dir
        )
        
        # Compare strategies
        compare_strategies([results["rsi"], results["ma_crossover"]], output_dir)
        
        return results
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def compare_strategies(results: List[Dict[str, Any]], output_dir: str = "backtest_results") -> None:
    """
    Compare multiple strategy backtest results.
    
    Args:
        results: List of backtest results
        output_dir: Directory to save comparison results
    """
    if not results:
        logger.warning("No results to compare")
        return
    
    logger.info("Comparing strategy performance...")
    
    # Create comparison DataFrame
    metrics = ["total_return", "annualized_return", "sharpe_ratio", "max_drawdown", "win_rate"]
    comparison = pd.DataFrame(index=metrics)
    
    for result in results:
        strategy_name = result["strategy"]
        comparison[strategy_name] = [result["metrics"][m] for m in metrics]
    
    # Log comparison
    logger.info("\nStrategy Comparison:")
    logger.info(f"\n{comparison}")
    
    # Save comparison to CSV
    output_path = Path(output_dir)
    comparison_file = output_path / "strategy_comparison.csv"
    comparison.to_csv(comparison_file)
    
    # Plot equity curves for comparison
    plt.figure(figsize=(12, 8))
    
    for result in results:
        equity_curve = result["equity_curve"]
        strategy_name = result["strategy"]
        plt.plot(equity_curve.index, equity_curve["equity"], label=strategy_name)
    
    plt.title("Strategy Comparison - Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True)
    
    # Save comparison plot
    comparison_plot_file = output_path / "plots" / "strategy_comparison.png"
    plt.savefig(comparison_plot_file)
    
    logger.info(f"Comparison plot saved to {comparison_plot_file}")


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logger
    setup_logger()
    
    # Parse symbols
    symbols = args.symbols.split(",")
    
    # Parse dates
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = datetime.now() - timedelta(days=365)  # Default to 1 year ago
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now()  # Default to today
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run continuous backtest
    run_continuous_backtest(
        strategy_name=args.strategy,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval=args.interval,
        initial_capital=args.initial_capital,
        commission_rate=args.commission_rate,
        slippage_rate=args.slippage_rate,
        output_dir=str(output_dir)
    )
    
    logger.success("Continuous backtest completed!")


if __name__ == "__main__":
    main()
