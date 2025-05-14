"""
KITE Trading System API

This module provides a FastAPI-based API for the KITE trading system,
allowing configuration and control of trading strategies through a web interface.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import threading
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Import KITE modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import core modules
from kite.core.models import BacktestConfig, StrategyConfig, Order, Portfolio
from kite.core.data import get_data_manager
from kite.utils.logging import setup_logger

# Import strategies
from kite.strategies.examples.rsi_strategy import RSIStrategy
from kite.strategies.examples.moving_average_crossover import MovingAverageCrossoverStrategy

# Import backtest engine
from kite.backtest.engine import BacktestEngine

# Initialize logger
setup_logger()

# Create FastAPI app
app = FastAPI(
    title="KITE Trading System API",
    description="API for configuring and controlling the KITE trading system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data storage
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CONFIG_DIR = Path("config")
CONFIG_DIR.mkdir(exist_ok=True)

# Store active simulations
active_simulations = {}
active_backtests = {}

# Define models
class StrategyParameters(BaseModel):
    """Base model for strategy parameters."""
    position_size: float = Field(0.1, ge=0.0, le=1.0, description="Position size as percentage of portfolio")


class RSIParameters(StrategyParameters):
    """Parameters for RSI strategy."""
    rsi_period: int = Field(14, ge=2, le=50, description="RSI period")
    oversold_level: float = Field(30.0, ge=10.0, le=50.0, description="Oversold level")
    overbought_level: float = Field(70.0, ge=50.0, le=90.0, description="Overbought level")


class MAParameters(StrategyParameters):
    """Parameters for Moving Average Crossover strategy."""
    fast_period: int = Field(20, ge=2, le=50, description="Fast MA period")
    slow_period: int = Field(50, ge=10, le=200, description="Slow MA period")
    signal_period: int = Field(9, ge=2, le=20, description="Signal period")


class BacktestRequest(BaseModel):
    """Request model for running a backtest."""
    strategy_type: str = Field(..., description="Strategy type (rsi or ma_crossover)")
    symbols: List[str] = Field(..., description="List of symbols to trade")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    interval: str = Field("1d", description="Data interval")
    initial_capital: float = Field(100000.0, ge=1000.0, description="Initial capital")
    commission_rate: float = Field(0.002, ge=0.0, le=0.1, description="Commission rate")
    slippage_rate: float = Field(0.0005, ge=0.0, le=0.01, description="Slippage rate")
    rsi_params: Optional[RSIParameters] = Field(None, description="RSI strategy parameters")
    ma_params: Optional[MAParameters] = Field(None, description="MA Crossover strategy parameters")


class SimulationRequest(BaseModel):
    """Request model for running a live simulation."""
    strategy_type: str = Field(..., description="Strategy type (rsi or ma_crossover)")
    symbols: List[str] = Field(..., description="List of symbols to trade")
    interval: str = Field("1d", description="Data interval")
    initial_capital: float = Field(100000.0, ge=1000.0, description="Initial capital")
    commission_rate: float = Field(0.002, ge=0.0, le=0.1, description="Commission rate")
    slippage_rate: float = Field(0.0005, ge=0.0, le=0.01, description="Slippage rate")
    history_days: int = Field(180, ge=30, le=365, description="Days of historical data")
    update_interval: int = Field(60, ge=10, le=3600, description="Update interval in seconds")
    rsi_params: Optional[RSIParameters] = Field(None, description="RSI strategy parameters")
    ma_params: Optional[MAParameters] = Field(None, description="MA Crossover strategy parameters")


class SimulationStatus(BaseModel):
    """Status of a simulation."""
    id: str
    strategy_type: str
    symbols: List[str]
    start_time: str
    status: str
    metrics: Dict[str, Any]


class BacktestStatus(BaseModel):
    """Status of a backtest."""
    id: str
    strategy_type: str
    symbols: List[str]
    start_date: str
    end_date: str
    status: str
    metrics: Optional[Dict[str, Any]]


# Helper functions
def create_strategy_config(strategy_type: str, symbols: List[str], params: Dict[str, Any]) -> StrategyConfig:
    """Create a strategy configuration."""
    if strategy_type == "rsi":
        return StrategyConfig(
            name="RSI Strategy",
            symbols=symbols,
            parameters=params
        )
    elif strategy_type == "ma_crossover":
        return StrategyConfig(
            name="MA Crossover",
            symbols=symbols,
            parameters=params
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


async def run_backtest_task(backtest_id: str, request: BacktestRequest):
    """Run a backtest in the background."""
    try:
        # Update status
        active_backtests[backtest_id]["status"] = "running"
        
        # Parse dates
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        # Get strategy parameters
        if request.strategy_type == "rsi":
            if not request.rsi_params:
                raise ValueError("RSI parameters are required for RSI strategy")
            params = request.rsi_params.dict()
        elif request.strategy_type == "ma_crossover":
            if not request.ma_params:
                raise ValueError("MA parameters are required for MA Crossover strategy")
            params = request.ma_params.dict()
        else:
            raise ValueError(f"Unknown strategy type: {request.strategy_type}")
        
        # Create strategy config
        strategy_config = create_strategy_config(request.strategy_type, request.symbols, params)
        
        # Create backtest config
        backtest_config = BacktestConfig(
            id=backtest_id,
            start_date=start_date,
            end_date=end_date,
            interval=request.interval,
            initial_capital=request.initial_capital,
            commission_rate=request.commission_rate,
            slippage_rate=request.slippage_rate
        )
        
        # Create backtest engine
        engine = BacktestEngine(strategy_config, backtest_config)
        
        # Run backtest
        result = engine.run()
        
        # Save results
        backtest_dir = RESULTS_DIR / f"backtest_{backtest_id}"
        backtest_dir.mkdir(exist_ok=True)
        
        # Save metrics
        metrics_file = backtest_dir / "metrics.json"
        metrics = {k: float(v) if hasattr(v, "item") else v for k, v in result.metrics.items()}
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Save equity curve
        equity_file = backtest_dir / "equity.csv"
        result.equity_curve.to_csv(equity_file)
        
        # Save trades
        trades_file = backtest_dir / "trades.csv"
        trades_df = pd.DataFrame([t.to_dict() for t in result.trades])
        if not trades_df.empty:
            trades_df.to_csv(trades_file, index=False)
        
        # Update status
        active_backtests[backtest_id]["status"] = "completed"
        active_backtests[backtest_id]["metrics"] = metrics
        
    except Exception as e:
        # Update status
        active_backtests[backtest_id]["status"] = "failed"
        active_backtests[backtest_id]["error"] = str(e)


class LiveSimulation:
    """Live trading simulation class."""
    
    def __init__(
        self,
        simulation_id: str,
        strategy_type: str,
        symbols: List[str],
        interval: str = "1d",
        initial_capital: float = 100000.0,
        commission_rate: float = 0.002,
        slippage_rate: float = 0.0005,
        history_days: int = 180,
        update_interval: int = 60,
        strategy_params: Dict[str, Any] = None
    ):
        """Initialize the live simulation."""
        self.simulation_id = simulation_id
        self.strategy_type = strategy_type
        self.symbols = symbols
        self.interval = interval
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.history_days = history_days
        self.update_interval = update_interval
        self.strategy_params = strategy_params or {}
        
        self.start_time = datetime.now()
        self.status = "initializing"
        self.is_running = False
        self.error = None
        self.metrics = {
            "total_return": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0
        }
        
        # Create output directory
        self.output_dir = RESULTS_DIR / f"simulation_{simulation_id}"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize strategy
        self.strategy_config = create_strategy_config(strategy_type, symbols, strategy_params)
        
        # Start simulation in a separate thread
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
    
    def _run(self):
        """Run the simulation."""
        try:
            # Initialize with historical data
            self._initialize()
            
            # Update status
            self.status = "running"
            self.is_running = True
            
            # Main simulation loop
            while self.is_running:
                # Update simulation
                self._update()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            self.is_running = False
    
    def _initialize(self):
        """Initialize the simulation with historical data."""
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
        if self.strategy_type == "rsi":
            self.strategy = RSIStrategy(self.strategy_config)
        elif self.strategy_type == "ma_crossover":
            self.strategy = MovingAverageCrossoverStrategy(self.strategy_config)
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")
        
        self.strategy.initialize(self.engine.data, self.initial_capital)
        
        # Run strategy on historical data
        self.strategy.run(self.engine.data)
        
        # Store initial state
        self.portfolio = self.strategy.portfolio
        self.positions = self.strategy.positions
        self.trades = self.strategy.trades
        self.equity_curve = self.strategy.equity_curve
        
        # Save initial state
        self._save_state()
    
    def _update(self):
        """Update the simulation with new data."""
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
        
        # Update portfolio and positions
        self.portfolio = self.strategy.portfolio
        self.positions = self.strategy.positions
        self.trades = self.strategy.trades
        self.equity_curve = self.strategy.equity_curve
        
        # Update metrics
        self._update_metrics()
        
        # Save state
        self._save_state()
    
    def _generate_new_bar(self):
        """Generate a new price bar for each symbol."""
        new_data = {}
        
        for symbol in self.symbols:
            # Get the last known data for this symbol
            last_data = self.engine.data[symbol].iloc[-1].copy()
            
            # Generate a new bar with some random price movement
            new_bar = pd.DataFrame(index=[datetime.now()])
            
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
    
    def _update_metrics(self):
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
    
    def _save_state(self):
        """Save the current state of the simulation."""
        # Save equity curve
        equity_file = self.output_dir / "equity.csv"
        self.equity_curve.to_csv(equity_file)
        
        # Save trades
        trades_file = self.output_dir / "trades.csv"
        trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
        if not trades_df.empty:
            trades_df.to_csv(trades_file, index=False)
        
        # Save metrics
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=4)
    
    def start(self):
        """Start the simulation."""
        if not self.thread.is_alive():
            self.thread.start()
    
    def stop(self):
        """Stop the simulation."""
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        self.status = "stopped"


# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to KITE Trading System API"}


@app.post("/backtest", response_model=BacktestStatus)
async def start_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Start a backtest."""
    # Generate a unique ID for this backtest
    backtest_id = str(uuid.uuid4())
    
    # Create backtest status
    backtest_status = {
        "id": backtest_id,
        "strategy_type": request.strategy_type,
        "symbols": request.symbols,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "status": "pending",
        "metrics": None
    }
    
    # Store backtest status
    active_backtests[backtest_id] = backtest_status
    
    # Start backtest in background
    background_tasks.add_task(run_backtest_task, backtest_id, request)
    
    return backtest_status


@app.get("/backtest/{backtest_id}", response_model=BacktestStatus)
async def get_backtest_status(backtest_id: str):
    """Get backtest status."""
    if backtest_id not in active_backtests:
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    return active_backtests[backtest_id]


@app.get("/backtests", response_model=List[BacktestStatus])
async def list_backtests():
    """List all backtests."""
    return list(active_backtests.values())


@app.post("/simulation", response_model=SimulationStatus)
async def start_simulation(request: SimulationRequest):
    """Start a live simulation."""
    # Generate a unique ID for this simulation
    simulation_id = str(uuid.uuid4())
    
    # Get strategy parameters
    if request.strategy_type == "rsi":
        if not request.rsi_params:
            raise HTTPException(status_code=400, detail="RSI parameters are required for RSI strategy")
        params = request.rsi_params.dict()
    elif request.strategy_type == "ma_crossover":
        if not request.ma_params:
            raise HTTPException(status_code=400, detail="MA parameters are required for MA Crossover strategy")
        params = request.ma_params.dict()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown strategy type: {request.strategy_type}")
    
    # Create simulation
    simulation = LiveSimulation(
        simulation_id=simulation_id,
        strategy_type=request.strategy_type,
        symbols=request.symbols,
        interval=request.interval,
        initial_capital=request.initial_capital,
        commission_rate=request.commission_rate,
        slippage_rate=request.slippage_rate,
        history_days=request.history_days,
        update_interval=request.update_interval,
        strategy_params=params
    )
    
    # Store simulation
    active_simulations[simulation_id] = simulation
    
    # Start simulation
    simulation.start()
    
    # Create simulation status
    simulation_status = {
        "id": simulation_id,
        "strategy_type": request.strategy_type,
        "symbols": request.symbols,
        "start_time": simulation.start_time.isoformat(),
        "status": simulation.status,
        "metrics": simulation.metrics
    }
    
    return simulation_status


@app.get("/simulation/{simulation_id}", response_model=SimulationStatus)
async def get_simulation_status(simulation_id: str):
    """Get simulation status."""
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    simulation = active_simulations[simulation_id]
    
    return {
        "id": simulation_id,
        "strategy_type": simulation.strategy_type,
        "symbols": simulation.symbols,
        "start_time": simulation.start_time.isoformat(),
        "status": simulation.status,
        "metrics": simulation.metrics
    }


@app.get("/simulations", response_model=List[SimulationStatus])
async def list_simulations():
    """List all simulations."""
    return [
        {
            "id": simulation_id,
            "strategy_type": simulation.strategy_type,
            "symbols": simulation.symbols,
            "start_time": simulation.start_time.isoformat(),
            "status": simulation.status,
            "metrics": simulation.metrics
        }
        for simulation_id, simulation in active_simulations.items()
    ]


@app.post("/simulation/{simulation_id}/stop")
async def stop_simulation(simulation_id: str):
    """Stop a simulation."""
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    simulation = active_simulations[simulation_id]
    simulation.stop()
    
    return {"message": f"Simulation {simulation_id} stopped"}


@app.get("/results/{result_id}/{file_name}")
async def get_result_file(result_id: str, file_name: str):
    """Get a result file."""
    # Check if it's a backtest or simulation
    if result_id.startswith("backtest_"):
        result_dir = RESULTS_DIR / result_id
    elif result_id.startswith("simulation_"):
        result_dir = RESULTS_DIR / result_id
    else:
        raise HTTPException(status_code=400, detail="Invalid result ID format")
    
    # Check if directory exists
    if not result_dir.exists():
        raise HTTPException(status_code=404, detail="Result directory not found")
    
    # Check if file exists
    file_path = result_dir / file_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)


@app.get("/strategies")
async def list_strategies():
    """List available strategies."""
    return [
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
    ]


@app.get("/symbols")
async def list_symbols(query: Optional[str] = None):
    """List available symbols."""
    # Use the data manager to search for symbols
    data_manager = get_data_manager()
    
    if query:
        symbols = data_manager.search_symbols(query)
    else:
        # Return a default list of popular symbols
        symbols = [
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
        ]
    
    return symbols


# Serve static files for frontend
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")


# Main function
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
