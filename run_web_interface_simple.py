#!/usr/bin/env python
"""
Simple Web Interface for KITE Trading System

This script provides a simplified web interface for configuring and running
trading strategies in the KITE system.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import webbrowser
import threading
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from flask import Flask, request, jsonify, render_template, send_from_directory
from kite.core.models import StrategyConfig, BacktestConfig
from kite.strategies.examples.rsi_strategy import RSIStrategy
from kite.strategies.examples.moving_average_crossover import MovingAverageCrossoverStrategy
from kite.backtest.engine import BacktestEngine
from kite.utils.logging import setup_logger
from loguru import logger

# Initialize Flask app
app = Flask(__name__, 
            static_folder=str(project_root / "frontend"),
            template_folder=str(project_root / "frontend"))

# Create necessary directories
os.makedirs(project_root / "results", exist_ok=True)

# Store active backtests and simulations
active_backtests = {}
active_simulations = {}

# Create sample backtest results for testing
def create_sample_backtest_results():
    """Create sample backtest results for testing."""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample backtest IDs
    sample_ids = [
        "rsi_20250512_123456",
        "ma_crossover_20250512_123457",
        "rsi_20250512_123458",
        "ma_crossover_20250512_123459"
    ]
    
    for backtest_id in sample_ids:
        # Create directory
        results_dir = project_root / "results" / backtest_id
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Create sample equity curve
        start_date = datetime.now() - timedelta(days=30)
        dates = [start_date + timedelta(days=i) for i in range(31)]
        
        # Generate random equity curve with upward trend
        initial_equity = 100000
        equity = [initial_equity]
        for i in range(1, 31):
            daily_return = np.random.normal(0.001, 0.01)  # Mean 0.1%, std 1%
            equity.append(equity[-1] * (1 + daily_return))
        
        # Calculate drawdown
        peak = equity[0]
        drawdown = [0]
        for i in range(1, 31):
            peak = max(peak, equity[i])
            drawdown.append((peak - equity[i]) / peak)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'equity': equity,
            'drawdown': drawdown
        })
        
        # Save equity curve - ensure proper formatting
        equity_file = results_dir / "equity.csv"
        # Convert timestamp to string format to avoid serialization issues
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df.to_csv(equity_file, index=False)
        
        # Create sample metrics
        metrics = {
            "total_return": (equity[-1] - equity[0]) / equity[0] * 100,
            "annualized_return": ((equity[-1] / equity[0]) ** (365 / 30) - 1) * 100,
            "sharpe_ratio": np.random.uniform(0.5, 2.5),
            "max_drawdown": max(drawdown) * 100,
            "win_rate": np.random.uniform(40, 60),
            "profit_factor": np.random.uniform(1.1, 2.0),
            "num_trades": np.random.randint(20, 100)
        }
        
        # Save metrics
        metrics_file = results_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Create sample trades
        num_trades = int(metrics["num_trades"])
        trades = []
        
        for i in range(num_trades):
            entry_date = start_date + timedelta(days=np.random.randint(0, 29))
            exit_date = entry_date + timedelta(days=np.random.randint(1, 3))
            
            is_long = np.random.random() > 0.5
            symbol = np.random.choice(["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"])
            entry_price = np.random.uniform(1000, 2000)
            exit_price = entry_price * (1 + np.random.normal(0.005, 0.02) * (1 if is_long else -1))
            quantity = np.random.randint(10, 100)
            pnl = (exit_price - entry_price) * quantity * (1 if is_long else -1)
            
            trades.append({
                "id": f"trade_{i}",
                "symbol": symbol,
                "direction": "long" if is_long else "short",
                "entry_date": entry_date.strftime("%Y-%m-%d %H:%M:%S"),
                "exit_date": exit_date.strftime("%Y-%m-%d %H:%M:%S"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "pnl": pnl
            })
        
        # Save trades
        trades_file = results_dir / "trades.json"
        with open(trades_file, "w") as f:
            json.dump(trades, f, indent=4)
        
        # Add to active backtests
        strategy_type = "rsi" if "rsi" in backtest_id else "ma_crossover"
        active_backtests[backtest_id] = {
            "id": backtest_id,
            "strategy_type": strategy_type,
            "symbols": ["RELIANCE", "TCS", "INFY"],
            "start_date": (start_date).strftime("%Y-%m-%d"),
            "end_date": (start_date + timedelta(days=30)).strftime("%Y-%m-%d"),
            "status": "completed",
            "metrics": metrics
        }
    
    logger.info(f"Created {len(sample_ids)} sample backtest results")
    return sample_ids

# Setup logger
setup_logger()

# Helper functions
def run_backtest(strategy_type, symbols, start_date, end_date, interval, initial_capital, 
                commission_rate, slippage_rate, strategy_params):
    """Run a backtest with the given parameters."""
    # Generate a unique ID for this backtest
    backtest_id = f"{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create strategy configuration
    strategy_config = StrategyConfig(
        name=f"{strategy_type.capitalize()} Strategy",
        symbols=symbols,
        parameters=strategy_params
    )
    
    # Parse dates
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Create backtest configuration
    backtest_config = BacktestConfig(
        id=backtest_id,
        strategy_name=strategy_config.name,
        start_date=start_date_obj,
        end_date=end_date_obj,
        initial_capital=initial_capital,
        symbols=symbols,
        interval=interval,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        strategy_parameters=strategy_params
    )
    
    # Create backtest engine
    engine = BacktestEngine(backtest_config)
    
    # Run backtest
    if strategy_type == "rsi":
        result = engine.run(RSIStrategy)
    elif strategy_type == "ma_crossover":
        result = engine.run(MovingAverageCrossoverStrategy)
    else:
        logger.error(f"Unknown strategy type: {strategy_type}")
        return None
    
    if result:
        # Save results
        results_dir = project_root / "results" / backtest_id
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Save metrics
        metrics_file = results_dir / "metrics.json"
        metrics = {k: float(v) if isinstance(v, (float, int)) else v for k, v in result.metrics.items()}
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Save equity curve
        equity_file = results_dir / "equity.csv"
        result.equity_curve.to_csv(equity_file)
        
        # Save trades
        if result.trades:
            trades_file = results_dir / "trades.csv"
            trades_data = [t.to_dict() for t in result.trades]
            with open(trades_file, "w") as f:
                json.dump(trades_data, f, indent=4)
        
        # Store result
        active_backtests[backtest_id] = {
            "id": backtest_id,
            "strategy_type": strategy_type,
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "status": "completed",
            "metrics": metrics
        }
        
        return active_backtests[backtest_id]
    else:
        logger.error("Backtest failed")
        return None

# Routes
@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/strategies')
def get_strategies():
    """Get available strategies."""
    return jsonify([
        {
            "id": "rsi",
            "name": "RSI Strategy",
            "description": "Relative Strength Index strategy",
            "parameters": {
                "rsi_period": {"type": "integer", "min": 2, "max": 50, "default": 14},
                "oversold_level": {"type": "number", "min": 10, "max": 50, "default": 30},
                "overbought_level": {"type": "number", "min": 50, "max": 90, "default": 70},
                "position_size": {"type": "number", "min": 0, "max": 1, "default": 0.1}
            }
        },
        {
            "id": "ma_crossover",
            "name": "Moving Average Crossover",
            "description": "Moving Average Crossover strategy",
            "parameters": {
                "fast_period": {"type": "integer", "min": 2, "max": 50, "default": 20},
                "slow_period": {"type": "integer", "min": 10, "max": 200, "default": 50},
                "signal_period": {"type": "integer", "min": 2, "max": 20, "default": 9},
                "position_size": {"type": "number", "min": 0, "max": 1, "default": 0.1}
            }
        }
    ])

@app.route('/api/symbols')
def get_symbols():
    """Get available symbols."""
    return jsonify([
        {"symbol": "RELIANCE", "name": "Reliance Industries Ltd", "exchange": "NSE"},
        {"symbol": "TCS", "name": "Tata Consultancy Services Ltd", "exchange": "NSE"},
        {"symbol": "INFY", "name": "Infosys Ltd", "exchange": "NSE"},
        {"symbol": "HDFCBANK", "name": "HDFC Bank Ltd", "exchange": "NSE"},
        {"symbol": "ICICIBANK", "name": "ICICI Bank Ltd", "exchange": "NSE"},
        {"symbol": "HDFC", "name": "Housing Development Finance Corporation Ltd", "exchange": "NSE"},
        {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank Ltd", "exchange": "NSE"},
        {"symbol": "ITC", "name": "ITC Ltd", "exchange": "NSE"},
        {"symbol": "SBIN", "name": "State Bank of India", "exchange": "NSE"},
        {"symbol": "BAJFINANCE", "name": "Bajaj Finance Ltd", "exchange": "NSE"}
    ])

@app.route('/api/backtest', methods=['POST'])
def start_backtest():
    """Start a backtest."""
    data = request.json
    
    # Extract parameters
    strategy_type = data.get('strategy_type')
    symbols = data.get('symbols', [])
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    interval = data.get('interval', '1d')
    initial_capital = float(data.get('initial_capital', 100000.0))
    commission_rate = float(data.get('commission_rate', 0.002))
    slippage_rate = float(data.get('slippage_rate', 0.0005))
    
    # Get strategy parameters
    if strategy_type == 'rsi':
        rsi_params = data.get('rsi_params', {})
        strategy_params = {
            'rsi_period': int(rsi_params.get('rsi_period', 14)),
            'oversold_level': float(rsi_params.get('oversold_level', 30.0)),
            'overbought_level': float(rsi_params.get('overbought_level', 70.0)),
            'position_size': float(rsi_params.get('position_size', 0.1))
        }
    elif strategy_type == 'ma_crossover':
        ma_params = data.get('ma_params', {})
        strategy_params = {
            'fast_period': int(ma_params.get('fast_period', 20)),
            'slow_period': int(ma_params.get('slow_period', 50)),
            'signal_period': int(ma_params.get('signal_period', 9)),
            'position_size': float(ma_params.get('position_size', 0.1))
        }
    else:
        return jsonify({"error": f"Unknown strategy type: {strategy_type}"}), 400
    
    # Run backtest in a separate thread
    def run_backtest_thread():
        result = run_backtest(
            strategy_type=strategy_type,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            strategy_params=strategy_params
        )
    
    # Create a unique ID for this backtest
    backtest_id = f"{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Store backtest status
    active_backtests[backtest_id] = {
        "id": backtest_id,
        "strategy_type": strategy_type,
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "status": "running",
        "metrics": None
    }
    
    # Start backtest thread
    threading.Thread(target=run_backtest_thread).start()
    
    return jsonify(active_backtests[backtest_id])

@app.route('/api/backtests')
def get_backtests():
    """Get all backtests."""
    return jsonify(list(active_backtests.values()))

@app.route('/api/backtest/<backtest_id>')
def get_backtest(backtest_id):
    """Get a specific backtest."""
    if backtest_id not in active_backtests:
        return jsonify({"error": "Backtest not found"}), 404
    
    return jsonify(active_backtests[backtest_id])

@app.route('/api/results/<result_id>/<file_name>')
def get_result_file(result_id, file_name):
    """Get a result file."""
    logger.info(f"Requested result file: {result_id}/{file_name}")
    
    # Ensure the results directory exists
    results_dir = project_root / "results" / result_id
    os.makedirs(results_dir, exist_ok=True)
    
    if not results_dir.exists():
        logger.error(f"Result directory not found: {results_dir}")
        return jsonify({"error": "Result directory not found"}), 404
    
    file_path = results_dir / file_name
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return jsonify({"error": "File not found"}), 404
    
    logger.info(f"Serving file: {file_path}")
    
    # Handle different file types appropriately
    if file_name.endswith('.csv'):
        # For CSV files, read and return as text
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            logger.info(f"Successfully read CSV file: {file_path}")
            return content, 200, {'Content-Type': 'text/csv'}
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 500
    elif file_name.endswith('.json'):
        # For JSON files, read and return as JSON
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
            logger.info(f"Successfully read JSON file: {file_path}")
            return jsonify(content)
        except Exception as e:
            logger.error(f"Error reading JSON file: {e}")
            return jsonify({"error": f"Error reading JSON file: {str(e)}"}), 500
    else:
        # For other files, let Flask determine the MIME type
        logger.info(f"Serving file with auto-detected MIME type: {file_path}")
        return send_from_directory(str(results_dir), file_name)

@app.route('/api/status')
def get_api_status():
    """Get API status."""
    return jsonify({
        "status": "connected",
        "angel_one": {
            "client_id": "AAAM356344",
            "trading_api": "ZNQY5zne",
            "historical_data_api": "10XN79Ba",
            "market_feeds_api": "nf3HXMX1",
            "status": "connected"
        }
    })

def open_browser():
    """Open the web interface in a browser."""
    # Wait for the server to start
    time.sleep(1)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    # Create sample backtest results for testing
    try:
        create_sample_backtest_results()
        print("Created sample backtest results for testing")
    except Exception as e:
        print(f"Error creating sample backtest results: {e}")
    
    # Open browser
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start server
    print("Starting KITE Trading System Web Interface at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, use_reloader=False)
