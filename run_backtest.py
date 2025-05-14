"""
Script to run a backtest using the KITE trading system.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from kite.core.models import StrategyConfig, BacktestConfig
from kite.strategies.examples.moving_average_crossover import MovingAverageCrossoverStrategy
from kite.strategies.examples.rsi_strategy import RSIStrategy
from kite.backtest.engine import BacktestEngine
from kite.analysis.plots import plot_equity_curve, plot_monthly_returns, plot_drawdown_periods
from kite.utils.logging import setup_logger
from loguru import logger

def run_ma_crossover_backtest():
    """Run a backtest for the Moving Average Crossover strategy."""
    logger.info("Running Moving Average Crossover strategy backtest...")
    
    # Create strategy configuration with more active parameters
    strategy_config = StrategyConfig(
        name="MA Crossover",
        symbols=["RELIANCE", "TCS", "INFY"],
        parameters={
            "fast_period": 10,  # Shorter fast period for more signals
            "slow_period": 30,  # Shorter slow period for more signals
            "signal_period": 5,  # Shorter signal period
            "position_size": 0.15  # Slightly larger position size
        }
    )
    
    # Create backtest configuration with shorter period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months instead of 1 year
    
    backtest_config = BacktestConfig(
        id="ma_crossover_test",
        strategy_name="MA Crossover",
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000.0,
        symbols=["RELIANCE", "TCS", "INFY"],
        interval="1d",
        commission_rate=0.0020,
        slippage_rate=0.0005,
        strategy_parameters=strategy_config.parameters
    )
    
    # Create backtest engine
    engine = BacktestEngine(backtest_config)
    
    # Run backtest
    result = engine.run(MovingAverageCrossoverStrategy)
    
    if result:
        logger.success("Backtest completed successfully!")
        
        # Print metrics
        logger.info("Performance Metrics:")
        for key, value in result.metrics.items():
            logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Create and save plots
        plots_dir = project_root / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Equity curve
        fig_equity = plot_equity_curve(result)
        fig_equity.savefig(plots_dir / "ma_crossover_equity.png")
        
        # Monthly returns
        fig_monthly = plot_monthly_returns(result)
        fig_monthly.savefig(plots_dir / "ma_crossover_monthly.png")
        
        # Drawdown periods
        fig_drawdown = plot_drawdown_periods(result)
        fig_drawdown.savefig(plots_dir / "ma_crossover_drawdown.png")
        
        logger.info(f"Plots saved to {plots_dir}")
        
        return result
    else:
        logger.error("Backtest failed!")
        return None

def run_rsi_backtest():
    """Run a backtest for the RSI strategy."""
    logger.info("Running RSI strategy backtest...")
    
    # Create strategy configuration with extremely aggressive parameters
    strategy_config = StrategyConfig(
        name="RSI Strategy",
        symbols=["RELIANCE", "TCS", "INFY"],
        parameters={
            "rsi_period": 8,  # Even shorter RSI period for more signals
            "oversold_level": 40,  # Much higher oversold level to ensure buy signals
            "overbought_level": 60,  # Much lower overbought level to ensure sell signals
            "position_size": 0.2  # Larger position size
        }
    )
    
    # Create backtest configuration with shorter period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months instead of 1 year
    
    backtest_config = BacktestConfig(
        id="rsi_test",
        strategy_name="RSI Strategy",
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000.0,
        symbols=["RELIANCE", "TCS", "INFY"],
        interval="1d",
        commission_rate=0.0020,
        slippage_rate=0.0005,
        strategy_parameters=strategy_config.parameters
    )
    
    # Create backtest engine
    engine = BacktestEngine(backtest_config)
    
    # Run backtest
    result = engine.run(RSIStrategy)
    
    if result:
        logger.success("Backtest completed successfully!")
        
        # Print metrics
        logger.info("Performance Metrics:")
        for key, value in result.metrics.items():
            logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Create and save plots
        plots_dir = project_root / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Equity curve
        fig_equity = plot_equity_curve(result)
        fig_equity.savefig(plots_dir / "rsi_equity.png")
        
        # Monthly returns
        fig_monthly = plot_monthly_returns(result)
        fig_monthly.savefig(plots_dir / "rsi_monthly.png")
        
        # Drawdown periods
        fig_drawdown = plot_drawdown_periods(result)
        fig_drawdown.savefig(plots_dir / "rsi_drawdown.png")
        
        logger.info(f"Plots saved to {plots_dir}")
        
        return result
    else:
        logger.error("Backtest failed!")
        return None

def compare_strategies(ma_result, rsi_result):
    """Compare the performance of both strategies."""
    if not ma_result or not rsi_result:
        logger.warning("Cannot compare strategies: one or both backtest results are missing.")
        return
    
    logger.info("Comparing strategy performance...")
    
    # Create DataFrame with metrics
    metrics = ["total_return", "annualized_return", "sharpe_ratio", "max_drawdown", "win_rate"]
    comparison = pd.DataFrame(index=metrics, columns=["MA Crossover", "RSI Strategy"])
    
    for metric in metrics:
        comparison.loc[metric, "MA Crossover"] = ma_result.metrics.get(metric, 0)
        comparison.loc[metric, "RSI Strategy"] = rsi_result.metrics.get(metric, 0)
    
    # Print comparison
    logger.info("\nStrategy Comparison:")
    logger.info(f"\n{comparison}")
    
    # Create and save comparison plot
    plots_dir = project_root / "plots"
    
    plt.figure(figsize=(10, 6))
    
    # Convert to percentage for better visualization
    for metric in ["total_return", "annualized_return", "max_drawdown"]:
        comparison.loc[metric] = comparison.loc[metric] * 100
    
    comparison.plot(kind="bar")
    plt.title("Strategy Comparison")
    plt.ylabel("Value (%)")
    plt.grid(axis="y")
    plt.tight_layout()
    
    plt.savefig(plots_dir / "strategy_comparison.png")
    logger.info(f"Comparison plot saved to {plots_dir / 'strategy_comparison.png'}")

def main():
    """Run all backtests and compare results."""
    # Setup logger
    setup_logger(level="INFO")
    
    logger.info("Starting KITE trading system backtest...")
    
    # Run backtests
    ma_result = run_ma_crossover_backtest()
    rsi_result = run_rsi_backtest()
    
    # Compare strategies
    compare_strategies(ma_result, rsi_result)
    
    logger.success("Backtest process completed!")

if __name__ == "__main__":
    main()
