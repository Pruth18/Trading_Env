"""
Plotting utilities for the KITE trading system.
Provides functions for visualizing trading strategy performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta

from kite.core.models import BacktestResult
from kite.backtest.metrics import (
    calculate_returns, calculate_drawdown, calculate_monthly_returns,
    calculate_rolling_metrics, calculate_all_metrics
)


def plot_equity_curve(backtest_result: BacktestResult, benchmark_data: Optional[pd.DataFrame] = None,
                      show_drawdown: bool = True, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot equity curve from a backtest result.
    
    Args:
        backtest_result: BacktestResult object
        benchmark_data: Optional DataFrame with benchmark data
        show_drawdown: Whether to show drawdown in a subplot
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    # Convert equity curve to DataFrame
    equity_curve = pd.DataFrame(backtest_result.equity_curve)
    
    # Set timestamp as index if available
    if 'timestamp' in equity_curve.columns:
        equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
        equity_curve.set_index('timestamp', inplace=True)
    
    # Calculate drawdown
    equity_curve = calculate_drawdown(equity_curve)
    
    # Create figure
    if show_drawdown:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    
    # Plot equity curve
    ax1.plot(equity_curve.index, equity_curve['equity'], label=backtest_result.config.strategy_name)
    
    # Plot benchmark if available
    if benchmark_data is not None:
        # Align benchmark data with equity curve
        benchmark_data = benchmark_data.reindex(equity_curve.index, method='ffill')
        
        # Normalize benchmark to start at the same value as the strategy
        benchmark_data = benchmark_data / benchmark_data.iloc[0] * equity_curve['equity'].iloc[0]
        
        # Plot benchmark
        ax1.plot(equity_curve.index, benchmark_data, label='Benchmark', alpha=0.7)
    
    # Format equity curve plot
    ax1.set_title(f"Equity Curve - {backtest_result.config.strategy_name}")
    ax1.set_ylabel("Portfolio Value")
    ax1.grid(True)
    ax1.legend()
    
    # Format x-axis
    if isinstance(equity_curve.index, pd.DatetimeIndex):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
    
    # Plot drawdown if requested
    if show_drawdown:
        # Plot drawdown
        ax2.fill_between(equity_curve.index, 0, equity_curve['drawdown'] * 100, color='red', alpha=0.3)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True)
        ax2.set_ylim(equity_curve['drawdown'].min() * 100 * 1.1, 0)  # Leave some space at the bottom
    
    plt.tight_layout()
    return fig


def plot_monthly_returns(backtest_result: BacktestResult, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot monthly returns heatmap from a backtest result.
    
    Args:
        backtest_result: BacktestResult object
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    # Convert equity curve to DataFrame
    equity_curve = pd.DataFrame(backtest_result.equity_curve)
    
    # Set timestamp as index if available
    if 'timestamp' in equity_curve.columns:
        equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
        equity_curve.set_index('timestamp', inplace=True)
    
    # Calculate monthly returns
    monthly_returns = calculate_monthly_returns(equity_curve)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(monthly_returns.iloc[:, :-1] * 100, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax)
    
    # Format plot
    ax.set_title(f"Monthly Returns (%) - {backtest_result.config.strategy_name}")
    
    plt.tight_layout()
    return fig


def plot_drawdown_periods(backtest_result: BacktestResult, top_n: int = 5, 
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot top drawdown periods from a backtest result.
    
    Args:
        backtest_result: BacktestResult object
        top_n: Number of top drawdown periods to show
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    # Convert equity curve to DataFrame
    equity_curve = pd.DataFrame(backtest_result.equity_curve)
    
    # Set timestamp as index if available
    if 'timestamp' in equity_curve.columns:
        equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
        equity_curve.set_index('timestamp', inplace=True)
    
    # Calculate drawdown
    equity_curve = calculate_drawdown(equity_curve)
    
    # Identify drawdown periods
    drawdown_periods = []
    current_drawdown = 0
    start_idx = None
    
    for i, row in equity_curve.iterrows():
        if row['drawdown'] < 0:
            if start_idx is None:
                start_idx = i
            current_drawdown = min(current_drawdown, row['drawdown'])
        elif start_idx is not None:
            # Drawdown ended
            end_idx = i
            duration = (end_idx - start_idx).days if isinstance(start_idx, pd.Timestamp) else (end_idx - start_idx)
            drawdown_periods.append({
                'start': start_idx,
                'end': end_idx,
                'drawdown': current_drawdown,
                'duration': duration
            })
            start_idx = None
            current_drawdown = 0
    
    # Add current drawdown if still ongoing
    if start_idx is not None:
        end_idx = equity_curve.index[-1]
        duration = (end_idx - start_idx).days if isinstance(start_idx, pd.Timestamp) else (end_idx - start_idx)
        drawdown_periods.append({
            'start': start_idx,
            'end': end_idx,
            'drawdown': current_drawdown,
            'duration': duration
        })
    
    # Sort drawdown periods by magnitude
    drawdown_periods.sort(key=lambda x: x['drawdown'])
    
    # Take top N drawdown periods
    top_drawdowns = drawdown_periods[:top_n]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot equity curve
    ax.plot(equity_curve.index, equity_curve['equity'], label='Equity', color='blue', alpha=0.5)
    
    # Highlight drawdown periods
    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(top_drawdowns)))
    
    for i, period in enumerate(top_drawdowns):
        start = period['start']
        end = period['end']
        drawdown = period['drawdown'] * 100
        duration = period['duration']
        
        # Highlight period
        ax.axvspan(start, end, alpha=0.3, color=colors[i])
        
        # Add annotation
        mid_point = start + (end - start) / 2
        y_pos = equity_curve.loc[start:end, 'equity'].min() * 0.95
        ax.annotate(f"{drawdown:.1f}% ({duration} days)", 
                   xy=(mid_point, y_pos),
                   xytext=(mid_point, y_pos * 0.9),
                   arrowprops=dict(arrowstyle="->", color='black'),
                   ha='center')
    
    # Format plot
    ax.set_title(f"Top {len(top_drawdowns)} Drawdown Periods - {backtest_result.config.strategy_name}")
    ax.set_ylabel("Portfolio Value")
    ax.set_xlabel("Date")
    ax.grid(True)
    
    # Format x-axis
    if isinstance(equity_curve.index, pd.DatetimeIndex):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig


def plot_rolling_metrics(backtest_result: BacktestResult, window: int = 252,
                         figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Plot rolling performance metrics from a backtest result.
    
    Args:
        backtest_result: BacktestResult object
        window: Rolling window size in days
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    # Convert equity curve to DataFrame
    equity_curve = pd.DataFrame(backtest_result.equity_curve)
    
    # Set timestamp as index if available
    if 'timestamp' in equity_curve.columns:
        equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
        equity_curve.set_index('timestamp', inplace=True)
    
    # Calculate rolling metrics
    rolling_metrics = calculate_rolling_metrics(equity_curve, window)
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    # Plot rolling return
    axes[0].plot(rolling_metrics.index, rolling_metrics['rolling_return'] * 100, color='blue')
    axes[0].set_title(f"Rolling {window}-Day Return (%)")
    axes[0].grid(True)
    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot rolling volatility
    axes[1].plot(rolling_metrics.index, rolling_metrics['rolling_volatility'] * 100, color='orange')
    axes[1].set_title(f"Rolling {window}-Day Volatility (%)")
    axes[1].grid(True)
    
    # Plot rolling Sharpe ratio
    axes[2].plot(rolling_metrics.index, rolling_metrics['rolling_sharpe'], color='green')
    axes[2].set_title(f"Rolling {window}-Day Sharpe Ratio")
    axes[2].grid(True)
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot rolling max drawdown
    axes[3].plot(rolling_metrics.index, rolling_metrics['rolling_max_drawdown'] * 100, color='red')
    axes[3].set_title(f"Rolling {window}-Day Max Drawdown (%)")
    axes[3].grid(True)
    axes[3].invert_yaxis()  # Invert y-axis so that larger drawdowns are lower
    
    # Format x-axis
    if isinstance(rolling_metrics.index, pd.DatetimeIndex):
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
    
    axes[3].set_xlabel("Date")
    
    plt.tight_layout()
    return fig


def plot_trade_analysis(backtest_result: BacktestResult, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Plot trade analysis from a backtest result.
    
    Args:
        backtest_result: BacktestResult object
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(backtest_result.trades)
    
    if trades_df.empty:
        # Create empty figure if no trades
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No trades to analyze", ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Convert timestamp to datetime
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    # Calculate profit/loss for each trade
    trades_df['pnl'] = np.where(
        trades_df['side'] == 'BUY',
        -trades_df['price'] * trades_df['quantity'] - trades_df['commission'],  # Buy: -price * quantity - commission
        trades_df['price'] * trades_df['quantity'] - trades_df['commission']    # Sell: price * quantity - commission
    )
    
    # Calculate cumulative P&L
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot cumulative P&L
    axes[0, 0].plot(trades_df['timestamp'], trades_df['cumulative_pnl'], marker='o', markersize=4)
    axes[0, 0].set_title("Cumulative P&L")
    axes[0, 0].grid(True)
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[0, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
    
    # Plot P&L distribution
    axes[0, 1].hist(trades_df['pnl'], bins=50, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--')
    axes[0, 1].set_title("P&L Distribution")
    axes[0, 1].grid(True)
    
    # Plot P&L by symbol
    if 'symbol' in trades_df.columns:
        symbol_pnl = trades_df.groupby('symbol')['pnl'].sum().sort_values()
        symbol_pnl.plot(kind='barh', ax=axes[1, 0])
        axes[1, 0].set_title("P&L by Symbol")
        axes[1, 0].grid(True)
    else:
        axes[1, 0].set_axis_off()
    
    # Plot trade count by month
    if 'timestamp' in trades_df.columns:
        trades_df['month'] = trades_df['timestamp'].dt.to_period('M')
        monthly_count = trades_df.groupby('month').size()
        monthly_count.index = monthly_count.index.astype(str)
        monthly_count.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title("Trade Count by Month")
        axes[1, 1].grid(True)
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
    else:
        axes[1, 1].set_axis_off()
    
    plt.tight_layout()
    return fig


def plot_interactive_equity_curve(backtest_result: BacktestResult, 
                                 benchmark_data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create an interactive equity curve plot using Plotly.
    
    Args:
        backtest_result: BacktestResult object
        benchmark_data: Optional DataFrame with benchmark data
        
    Returns:
        Plotly Figure object
    """
    # Convert equity curve to DataFrame
    equity_curve = pd.DataFrame(backtest_result.equity_curve)
    
    # Set timestamp as index if available
    if 'timestamp' in equity_curve.columns:
        equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
        equity_curve.set_index('timestamp', inplace=True)
    
    # Calculate drawdown
    equity_curve = calculate_drawdown(equity_curve)
    
    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"Equity Curve - {backtest_result.config.strategy_name}", "Drawdown (%)"))
    
    # Add equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index, 
            y=equity_curve['equity'],
            mode='lines',
            name=backtest_result.config.strategy_name,
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Add benchmark if available
    if benchmark_data is not None:
        # Align benchmark data with equity curve
        benchmark_data = benchmark_data.reindex(equity_curve.index, method='ffill')
        
        # Normalize benchmark to start at the same value as the strategy
        benchmark_data = benchmark_data / benchmark_data.iloc[0] * equity_curve['equity'].iloc[0]
        
        # Add benchmark to plot
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index, 
                y=benchmark_data,
                mode='lines',
                name='Benchmark',
                line=dict(color='gray', dash='dash')
            ),
            row=1, col=1
        )
    
    # Add drawdown
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index, 
            y=equity_curve['drawdown'] * 100,
            mode='lines',
            name='Drawdown',
            line=dict(color='red'),
            fill='tozeroy'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"Equity Curve and Drawdown - {backtest_result.config.strategy_name}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    # Update x-axis label
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    return fig


def plot_interactive_monthly_returns(backtest_result: BacktestResult) -> go.Figure:
    """
    Create an interactive monthly returns heatmap using Plotly.
    
    Args:
        backtest_result: BacktestResult object
        
    Returns:
        Plotly Figure object
    """
    # Convert equity curve to DataFrame
    equity_curve = pd.DataFrame(backtest_result.equity_curve)
    
    # Set timestamp as index if available
    if 'timestamp' in equity_curve.columns:
        equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
        equity_curve.set_index('timestamp', inplace=True)
    
    # Calculate monthly returns
    monthly_returns = calculate_monthly_returns(equity_curve)
    
    # Convert to format suitable for heatmap
    monthly_returns = monthly_returns.iloc[:, :-1] * 100  # Exclude annual column and convert to percentage
    
    # Create z values (returns) and text (formatted returns)
    z = monthly_returns.values
    text = np.array([[f"{val:.2f}%" for val in row] for row in z])
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=monthly_returns.columns,
        y=monthly_returns.index,
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Return (%)"),
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{text}<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Monthly Returns (%) - {backtest_result.config.strategy_name}",
        xaxis=dict(title="Month"),
        yaxis=dict(title="Year"),
        template="plotly_white"
    )
    
    return fig


def plot_interactive_trade_analysis(backtest_result: BacktestResult) -> Dict[str, go.Figure]:
    """
    Create interactive trade analysis plots using Plotly.
    
    Args:
        backtest_result: BacktestResult object
        
    Returns:
        Dictionary of Plotly Figure objects
    """
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(backtest_result.trades)
    
    if trades_df.empty:
        # Create empty figure if no trades
        fig = go.Figure()
        fig.add_annotation(
            text="No trades to analyze",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return {"no_trades": fig}
    
    # Convert timestamp to datetime
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    # Calculate profit/loss for each trade
    trades_df['pnl'] = np.where(
        trades_df['side'] == 'BUY',
        -trades_df['price'] * trades_df['quantity'] - trades_df['commission'],  # Buy: -price * quantity - commission
        trades_df['price'] * trades_df['quantity'] - trades_df['commission']    # Sell: price * quantity - commission
    )
    
    # Calculate cumulative P&L
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    
    # Create figures dictionary
    figures = {}
    
    # Cumulative P&L
    fig_cum_pnl = go.Figure()
    fig_cum_pnl.add_trace(go.Scatter(
        x=trades_df['timestamp'],
        y=trades_df['cumulative_pnl'],
        mode='lines+markers',
        name='Cumulative P&L',
        marker=dict(size=6)
    ))
    
    fig_cum_pnl.update_layout(
        title="Cumulative P&L",
        xaxis_title="Date",
        yaxis_title="P&L",
        template="plotly_white"
    )
    
    figures["cumulative_pnl"] = fig_cum_pnl
    
    # P&L Distribution
    fig_pnl_dist = go.Figure()
    fig_pnl_dist.add_trace(go.Histogram(
        x=trades_df['pnl'],
        nbinsx=50,
        marker_color='skyblue',
        marker_line=dict(color='black', width=1)
    ))
    
    fig_pnl_dist.add_shape(
        type="line",
        x0=0, y0=0,
        x1=0, y1=trades_df['pnl'].value_counts().max(),
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig_pnl_dist.update_layout(
        title="P&L Distribution",
        xaxis_title="P&L",
        yaxis_title="Frequency",
        template="plotly_white"
    )
    
    figures["pnl_distribution"] = fig_pnl_dist
    
    # P&L by Symbol
    if 'symbol' in trades_df.columns:
        symbol_pnl = trades_df.groupby('symbol')['pnl'].sum().sort_values()
        
        fig_symbol_pnl = go.Figure()
        fig_symbol_pnl.add_trace(go.Bar(
            y=symbol_pnl.index,
            x=symbol_pnl.values,
            orientation='h',
            marker_color=['red' if x < 0 else 'green' for x in symbol_pnl.values]
        ))
        
        fig_symbol_pnl.update_layout(
            title="P&L by Symbol",
            xaxis_title="P&L",
            yaxis_title="Symbol",
            template="plotly_white"
        )
        
        figures["pnl_by_symbol"] = fig_symbol_pnl
    
    # Trade Count by Month
    if 'timestamp' in trades_df.columns:
        trades_df['month'] = trades_df['timestamp'].dt.to_period('M').astype(str)
        monthly_count = trades_df.groupby('month').size()
        
        fig_monthly_count = go.Figure()
        fig_monthly_count.add_trace(go.Bar(
            x=monthly_count.index,
            y=monthly_count.values
        ))
        
        fig_monthly_count.update_layout(
            title="Trade Count by Month",
            xaxis_title="Month",
            yaxis_title="Number of Trades",
            template="plotly_white",
            xaxis=dict(tickangle=45)
        )
        
        figures["trade_count_by_month"] = fig_monthly_count
    
    return figures


def create_performance_dashboard(backtest_result: BacktestResult,
                                benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, go.Figure]:
    """
    Create a complete performance dashboard using Plotly.
    
    Args:
        backtest_result: BacktestResult object
        benchmark_data: Optional DataFrame with benchmark data
        
    Returns:
        Dictionary of Plotly Figure objects
    """
    # Calculate metrics
    metrics = calculate_all_metrics(backtest_result)
    
    # Create dashboard figures
    dashboard = {}
    
    # Equity curve and drawdown
    dashboard["equity_curve"] = plot_interactive_equity_curve(backtest_result, benchmark_data)
    
    # Monthly returns heatmap
    dashboard["monthly_returns"] = plot_interactive_monthly_returns(backtest_result)
    
    # Trade analysis
    trade_figures = plot_interactive_trade_analysis(backtest_result)
    dashboard.update(trade_figures)
    
    # Metrics summary
    fig_metrics = go.Figure(data=[go.Table(
        header=dict(
            values=["Metric", "Value"],
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[
                list(metrics.keys()),
                [f"{v:.4f}" if isinstance(v, float) else v for v in metrics.values()]
            ],
            fill_color='lavender',
            align='left'
        )
    )])
    
    fig_metrics.update_layout(
        title="Performance Metrics",
        template="plotly_white"
    )
    
    dashboard["metrics_summary"] = fig_metrics
    
    return dashboard
