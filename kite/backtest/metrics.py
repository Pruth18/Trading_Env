"""
Performance metrics calculation for the KITE trading system.
Provides functions for calculating and analyzing trading strategy performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from kite.core.models import BacktestResult


def calculate_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate returns from an equity curve.
    
    Args:
        equity_curve: DataFrame with equity values
        
    Returns:
        DataFrame with returns added
    """
    df = equity_curve.copy()
    
    # Calculate returns
    df['return'] = df['equity'].pct_change()
    df['log_return'] = np.log(df['equity'] / df['equity'].shift(1))
    
    # Calculate cumulative returns
    df['cum_return'] = (1 + df['return']).cumprod() - 1
    df['cum_log_return'] = df['log_return'].cumsum()
    
    return df


def calculate_drawdown(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate drawdown from an equity curve.
    
    Args:
        equity_curve: DataFrame with equity values
        
    Returns:
        DataFrame with drawdown metrics added
    """
    df = equity_curve.copy()
    
    # Calculate drawdown
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
    
    # Calculate drawdown duration
    df['is_drawdown'] = df['drawdown'] < 0
    df['drawdown_group'] = (df['is_drawdown'] != df['is_drawdown'].shift()).cumsum()
    
    # Calculate underwater duration (days in drawdown)
    df['underwater_days'] = df.groupby('drawdown_group').cumcount()
    df.loc[~df['is_drawdown'], 'underwater_days'] = 0
    
    return df


def calculate_basic_metrics(equity_curve: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate basic performance metrics from an equity curve.
    
    Args:
        equity_curve: DataFrame with equity values
        
    Returns:
        Dictionary of performance metrics
    """
    # Calculate returns
    df = calculate_returns(equity_curve)
    
    # Calculate drawdown
    df = calculate_drawdown(df)
    
    # Basic metrics
    initial_equity = df['equity'].iloc[0]
    final_equity = df['equity'].iloc[-1]
    
    # Return metrics
    total_return = (final_equity / initial_equity) - 1
    
    # Annualized return
    days = (df.index[-1] - df.index[0]).days if isinstance(df.index, pd.DatetimeIndex) else len(df)
    years = days / 365.25
    annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1
    
    # Volatility
    daily_returns = df['return'].dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    
    # Drawdown metrics
    max_drawdown = df['drawdown'].min()
    max_drawdown_duration = df['underwater_days'].max()
    
    # Risk metrics
    risk_free_rate = 0.02  # 2% annual risk-free rate
    daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = daily_returns - daily_risk_free
    
    # Sharpe ratio
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    # Sortino ratio
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
    
    # Return metrics
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_drawdown_duration,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio
    }
    
    return metrics


def calculate_trade_metrics(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate trade-based performance metrics.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Dictionary of trade metrics
    """
    if not trades:
        return {
            'num_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'avg_trade': 0,
            'max_win': 0,
            'max_loss': 0
        }
    
    # Convert to DataFrame
    df = pd.DataFrame(trades)
    
    # Calculate profit/loss for each trade
    df['pnl'] = np.where(
        df['side'] == 'BUY',
        -df['price'] * df['quantity'] - df['commission'],  # Buy: -price * quantity - commission
        df['price'] * df['quantity'] - df['commission']    # Sell: price * quantity - commission
    )
    
    # Basic trade metrics
    num_trades = len(df)
    winning_trades = df[df['pnl'] > 0]
    losing_trades = df[df['pnl'] <= 0]
    
    win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
    
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    
    profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if losing_trades['pnl'].sum() != 0 else float('inf')
    
    avg_trade = df['pnl'].mean()
    max_win = winning_trades['pnl'].max() if len(winning_trades) > 0 else 0
    max_loss = losing_trades['pnl'].min() if len(losing_trades) > 0 else 0
    
    # Return metrics
    metrics = {
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'avg_trade': avg_trade,
        'max_win': max_win,
        'max_loss': max_loss
    }
    
    return metrics


def calculate_advanced_metrics(equity_curve: pd.DataFrame, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Calculate advanced performance metrics from an equity curve.
    
    Args:
        equity_curve: DataFrame with equity values
        benchmark_returns: Optional Series of benchmark returns
        
    Returns:
        Dictionary of advanced performance metrics
    """
    # Calculate returns
    df = calculate_returns(equity_curve)
    
    # Calculate drawdown
    df = calculate_drawdown(df)
    
    # Daily returns
    daily_returns = df['return'].dropna()
    
    # Risk-free rate
    risk_free_rate = 0.02  # 2% annual risk-free rate
    daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
    
    # Excess returns
    excess_returns = daily_returns - daily_risk_free
    
    # Downside returns
    downside_returns = daily_returns[daily_returns < 0]
    
    # Upside returns
    upside_returns = daily_returns[daily_returns > 0]
    
    # Benchmark metrics
    alpha = 0.0
    beta = 0.0
    r_squared = 0.0
    tracking_error = 0.0
    information_ratio = 0.0
    
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # Align benchmark returns with strategy returns
        benchmark_returns = benchmark_returns.reindex(daily_returns.index)
        
        # Calculate beta
        covariance = np.cov(daily_returns, benchmark_returns)[0, 1]
        variance = np.var(benchmark_returns)
        beta = covariance / variance if variance > 0 else 0
        
        # Calculate alpha
        alpha = daily_returns.mean() - beta * benchmark_returns.mean()
        
        # Calculate R-squared
        correlation = np.corrcoef(daily_returns, benchmark_returns)[0, 1]
        r_squared = correlation ** 2
        
        # Calculate tracking error
        tracking_error = np.std(daily_returns - benchmark_returns) * np.sqrt(252)
        
        # Calculate information ratio
        information_ratio = (daily_returns.mean() - benchmark_returns.mean()) / tracking_error if tracking_error > 0 else 0
    
    # Kurtosis and skewness
    kurtosis = daily_returns.kurtosis()
    skewness = daily_returns.skew()
    
    # Value at Risk (VaR)
    var_95 = np.percentile(daily_returns, 5)
    var_99 = np.percentile(daily_returns, 1)
    
    # Conditional Value at Risk (CVaR) / Expected Shortfall
    cvar_95 = daily_returns[daily_returns <= var_95].mean()
    cvar_99 = daily_returns[daily_returns <= var_99].mean()
    
    # Omega ratio
    omega_ratio = upside_returns.sum() / abs(downside_returns.sum()) if abs(downside_returns.sum()) > 0 else float('inf')
    
    # Return metrics
    metrics = {
        'alpha': alpha * 252,  # Annualized
        'beta': beta,
        'r_squared': r_squared,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio,
        'kurtosis': kurtosis,
        'skewness': skewness,
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'omega_ratio': omega_ratio
    }
    
    return metrics


def calculate_all_metrics(backtest_result: BacktestResult, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Calculate all performance metrics for a backtest result.
    
    Args:
        backtest_result: BacktestResult object
        benchmark_returns: Optional Series of benchmark returns
        
    Returns:
        Dictionary of all performance metrics
    """
    # Convert equity curve to DataFrame
    equity_curve = pd.DataFrame(backtest_result.equity_curve)
    
    # Set timestamp as index if available
    if 'timestamp' in equity_curve.columns:
        equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
        equity_curve.set_index('timestamp', inplace=True)
    
    # Calculate basic metrics
    basic_metrics = calculate_basic_metrics(equity_curve)
    
    # Calculate trade metrics
    trade_metrics = calculate_trade_metrics(backtest_result.trades)
    
    # Calculate advanced metrics
    advanced_metrics = calculate_advanced_metrics(equity_curve, benchmark_returns)
    
    # Combine all metrics
    all_metrics = {**basic_metrics, **trade_metrics, **advanced_metrics}
    
    return all_metrics


def compare_strategies(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """
    Compare multiple strategy backtest results.
    
    Args:
        results: Dictionary of strategy name to BacktestResult
        
    Returns:
        DataFrame with comparison metrics
    """
    # Calculate metrics for each strategy
    metrics = {}
    for strategy_name, result in results.items():
        metrics[strategy_name] = calculate_all_metrics(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics)
    
    return df


def calculate_monthly_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly returns from an equity curve.
    
    Args:
        equity_curve: DataFrame with equity values
        
    Returns:
        DataFrame with monthly returns
    """
    # Convert equity curve to DataFrame
    df = equity_curve.copy()
    
    # Set timestamp as index if available
    if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    # Calculate returns
    df['return'] = df['equity'].pct_change()
    
    # Resample to monthly returns
    monthly_returns = df['return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create a DataFrame with years as rows and months as columns
    result = monthly_returns.to_frame()
    result['year'] = result.index.year
    result['month'] = result.index.month
    result = result.pivot(index='year', columns='month', values='return')
    
    # Add annual return column
    annual_returns = df['return'].resample('A').apply(lambda x: (1 + x).prod() - 1)
    result['Annual'] = annual_returns.values
    
    # Rename columns to month names
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    result.rename(columns=month_names, inplace=True)
    
    return result


def calculate_rolling_metrics(equity_curve: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Calculate rolling performance metrics from an equity curve.
    
    Args:
        equity_curve: DataFrame with equity values
        window: Rolling window size in days
        
    Returns:
        DataFrame with rolling metrics
    """
    # Convert equity curve to DataFrame
    df = equity_curve.copy()
    
    # Set timestamp as index if available
    if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    # Calculate returns
    df['return'] = df['equity'].pct_change()
    
    # Calculate rolling metrics
    df['rolling_return'] = (1 + df['return']).rolling(window).prod() - 1
    df['rolling_volatility'] = df['return'].rolling(window).std() * np.sqrt(252)
    df['rolling_sharpe'] = df['rolling_return'] / df['rolling_volatility'] if 'rolling_volatility' in df else 0
    df['rolling_max_drawdown'] = df['equity'].rolling(window).apply(lambda x: (x / x.cummax() - 1).min())
    
    return df[['rolling_return', 'rolling_volatility', 'rolling_sharpe', 'rolling_max_drawdown']]
