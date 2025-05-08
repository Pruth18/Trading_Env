"""
Web server for the KITE trading system.
Provides a web interface for managing strategies, running backtests, and viewing results.
"""

import os
import sys
import json
import uuid
import importlib
import inspect
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Type
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from kite.core.models import (
    StrategyConfig, BacktestConfig, BacktestResult, StrategyType
)
from kite.core.data import get_data_manager
from kite.strategies.base import Strategy
from kite.backtest.engine import BacktestEngine
from kite.utils.config import Config, ROOT_DIR
from kite.utils.logging import logger


# Create FastAPI app
app = FastAPI(
    title="KITE Trading System",
    description="A modular and scalable trading environment for backtesting and deploying trading strategies.",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = ROOT_DIR / "kite" / "app" / "ui" / "static"
static_dir.mkdir(exist_ok=True, parents=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Store running backtests
running_backtests: Dict[str, Dict[str, Any]] = {}


# Pydantic models for API
class StrategyRequest(BaseModel):
    name: str
    type: str = "CUSTOM"
    description: str = ""
    symbols: List[str]
    parameters: Dict[str, Any] = {}
    code: Optional[str] = None


class BacktestRequest(BaseModel):
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    symbols: List[str]
    interval: str = "1d"
    commission_rate: float = 0.0020
    slippage_rate: float = 0.0005
    parameters: Dict[str, Any] = {}


class BacktestStatusResponse(BaseModel):
    id: str
    strategy_name: str
    status: str
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result_id: Optional[str] = None
    error: Optional[str] = None


@app.get("/")
async def get_index():
    """Serve the main dashboard page."""
    index_path = ROOT_DIR / "kite" / "app" / "ui" / "index.html"
    
    if not index_path.exists():
        # Create a simple index page if it doesn't exist
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>KITE Trading System</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { padding-top: 20px; }
                .navbar { margin-bottom: 20px; }
                .card { margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>KITE Trading System</h1>
                <p>Welcome to the KITE Trading System dashboard. Please use the API to interact with the system.</p>
                <p>API documentation is available at <a href="/docs">/docs</a>.</p>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    return FileResponse(index_path)


@app.get("/api/strategies", response_model=List[Dict[str, Any]])
async def get_strategies():
    """Get all available strategies."""
    try:
        # Get strategy configs from config directory
        strategies = Config.list_strategy_configs()
        
        # Convert to list of dictionaries
        strategy_list = []
        for name, config in strategies.items():
            strategy_list.append({
                "name": name,
                "type": config.get("type", "CUSTOM"),
                "description": config.get("description", ""),
                "symbols": config.get("symbols", []),
                "parameters": config.get("parameters", {}),
                "enabled": config.get("enabled", True)
            })
        
        return strategy_list
    except Exception as e:
        logger.error(f"Error getting strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/strategies", response_model=Dict[str, Any])
async def create_strategy(strategy: StrategyRequest):
    """Create a new strategy."""
    try:
        # Check if strategy already exists
        existing_config = Config.load_strategy_config(strategy.name)
        if existing_config:
            raise HTTPException(status_code=400, detail=f"Strategy '{strategy.name}' already exists")
        
        # Create strategy config
        config = {
            "name": strategy.name,
            "type": strategy.type,
            "description": strategy.description,
            "symbols": strategy.symbols,
            "parameters": strategy.parameters,
            "enabled": True
        }
        
        # Save strategy config
        Config.save_strategy_config(strategy.name, config)
        
        # If code is provided, save it to a Python file
        if strategy.code:
            # Create strategies directory if it doesn't exist
            strategies_dir = ROOT_DIR / "kite" / "strategies"
            strategies_dir.mkdir(exist_ok=True, parents=True)
            
            # Save strategy code to file
            strategy_file = strategies_dir / f"{strategy.name.lower()}.py"
            with open(strategy_file, "w") as f:
                f.write(strategy.code)
            
            logger.info(f"Saved strategy code to {strategy_file}")
        
        logger.info(f"Created strategy '{strategy.name}'")
        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategies/{name}", response_model=Dict[str, Any])
async def get_strategy(name: str):
    """Get a strategy by name."""
    try:
        # Load strategy config
        config = Config.load_strategy_config(name)
        if not config:
            raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")
        
        # Check if strategy code file exists
        strategy_file = ROOT_DIR / "kite" / "strategies" / f"{name.lower()}.py"
        if strategy_file.exists():
            with open(strategy_file, "r") as f:
                config["code"] = f.read()
        
        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy '{name}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/strategies/{name}", response_model=Dict[str, Any])
async def update_strategy(name: str, strategy: StrategyRequest):
    """Update an existing strategy."""
    try:
        # Check if strategy exists
        existing_config = Config.load_strategy_config(name)
        if not existing_config:
            raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")
        
        # Update strategy config
        config = {
            "name": strategy.name,
            "type": strategy.type,
            "description": strategy.description,
            "symbols": strategy.symbols,
            "parameters": strategy.parameters,
            "enabled": existing_config.get("enabled", True)
        }
        
        # Save strategy config
        Config.save_strategy_config(name, config)
        
        # If code is provided, save it to a Python file
        if strategy.code:
            # Create strategies directory if it doesn't exist
            strategies_dir = ROOT_DIR / "kite" / "strategies"
            strategies_dir.mkdir(exist_ok=True, parents=True)
            
            # Save strategy code to file
            strategy_file = strategies_dir / f"{name.lower()}.py"
            with open(strategy_file, "w") as f:
                f.write(strategy.code)
            
            logger.info(f"Updated strategy code in {strategy_file}")
        
        logger.info(f"Updated strategy '{name}'")
        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating strategy '{name}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/strategies/{name}", response_model=Dict[str, str])
async def delete_strategy(name: str):
    """Delete a strategy."""
    try:
        # Check if strategy exists
        config_file = ROOT_DIR / "config" / f"{name}.json"
        if not config_file.exists():
            raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")
        
        # Delete strategy config file
        config_file.unlink()
        
        # Delete strategy code file if it exists
        strategy_file = ROOT_DIR / "kite" / "strategies" / f"{name.lower()}.py"
        if strategy_file.exists():
            strategy_file.unlink()
        
        logger.info(f"Deleted strategy '{name}'")
        return {"message": f"Strategy '{name}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting strategy '{name}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest", response_model=Dict[str, str])
async def run_backtest(backtest: BacktestRequest, background_tasks: BackgroundTasks):
    """Run a backtest for a strategy."""
    try:
        # Check if strategy exists
        strategy_config = Config.load_strategy_config(backtest.strategy_name)
        if not strategy_config:
            raise HTTPException(status_code=404, detail=f"Strategy '{backtest.strategy_name}' not found")
        
        # Generate backtest ID
        backtest_id = str(uuid.uuid4())
        
        # Create backtest config
        config = BacktestConfig(
            id=backtest_id,
            strategy_name=backtest.strategy_name,
            start_date=backtest.start_date,
            end_date=backtest.end_date,
            initial_capital=backtest.initial_capital,
            symbols=backtest.symbols,
            interval=backtest.interval,
            commission_rate=backtest.commission_rate,
            slippage_rate=backtest.slippage_rate,
            strategy_parameters=backtest.parameters
        )
        
        # Store backtest info
        running_backtests[backtest_id] = {
            "id": backtest_id,
            "strategy_name": backtest.strategy_name,
            "status": "QUEUED",
            "progress": 0.0,
            "start_time": None,
            "end_time": None,
            "result_id": None,
            "error": None,
            "config": config
        }
        
        # Run backtest in background
        background_tasks.add_task(_run_backtest_task, backtest_id, config)
        
        logger.info(f"Queued backtest '{backtest_id}' for strategy '{backtest.strategy_name}'")
        return {"id": backtest_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_backtest_task(backtest_id: str, config: BacktestConfig):
    """Background task to run a backtest."""
    try:
        # Update status
        running_backtests[backtest_id]["status"] = "RUNNING"
        running_backtests[backtest_id]["start_time"] = datetime.now()
        
        logger.info(f"Starting backtest '{backtest_id}' for strategy '{config.strategy_name}'")
        
        # Load strategy class
        strategy_class = _load_strategy_class(config.strategy_name)
        if not strategy_class:
            raise ValueError(f"Strategy class for '{config.strategy_name}' not found")
        
        # Create backtest engine
        engine = BacktestEngine(config)
        
        # Run backtest
        result = engine.run(strategy_class)
        
        if not result:
            raise ValueError("Backtest failed to produce a result")
        
        # Save backtest result
        result_id = _save_backtest_result(result)
        
        # Update status
        running_backtests[backtest_id]["status"] = "COMPLETED"
        running_backtests[backtest_id]["progress"] = 1.0
        running_backtests[backtest_id]["end_time"] = datetime.now()
        running_backtests[backtest_id]["result_id"] = result_id
        
        logger.info(f"Completed backtest '{backtest_id}' for strategy '{config.strategy_name}'")
        
    except Exception as e:
        logger.error(f"Error in backtest '{backtest_id}': {str(e)}")
        
        # Update status
        running_backtests[backtest_id]["status"] = "FAILED"
        running_backtests[backtest_id]["end_time"] = datetime.now()
        running_backtests[backtest_id]["error"] = str(e)


def _load_strategy_class(strategy_name: str) -> Optional[Type[Strategy]]:
    """Load a strategy class by name."""
    try:
        # First, check if there's a custom strategy file
        strategy_file = ROOT_DIR / "kite" / "strategies" / f"{strategy_name.lower()}.py"
        if strategy_file.exists():
            # Add parent directory to path if needed
            parent_dir = str(ROOT_DIR)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Import module
            module_name = f"kite.strategies.{strategy_name.lower()}"
            module = importlib.import_module(module_name)
            
            # Find strategy class
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and issubclass(obj, Strategy) and obj != Strategy):
                    return obj
        
        # If not found, check example strategies
        examples_dir = ROOT_DIR / "kite" / "strategies" / "examples"
        if examples_dir.exists():
            for file in examples_dir.glob("*.py"):
                if file.stem.lower() == strategy_name.lower():
                    # Import module
                    module_name = f"kite.strategies.examples.{file.stem}"
                    module = importlib.import_module(module_name)
                    
                    # Find strategy class
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and issubclass(obj, Strategy) and obj != Strategy):
                            return obj
        
        logger.warning(f"Strategy class for '{strategy_name}' not found")
        return None
        
    except Exception as e:
        logger.error(f"Error loading strategy class '{strategy_name}': {str(e)}")
        return None


def _save_backtest_result(result: BacktestResult) -> str:
    """Save a backtest result to file and return its ID."""
    try:
        # Create results directory if it doesn't exist
        results_dir = ROOT_DIR / "results"
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Save result to file
        result_id = result.id
        result_file = results_dir / f"{result_id}.json"
        
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=4, default=str)
        
        logger.info(f"Saved backtest result to {result_file}")
        return result_id
        
    except Exception as e:
        logger.error(f"Error saving backtest result: {str(e)}")
        raise


@app.get("/api/backtest/{backtest_id}/status", response_model=BacktestStatusResponse)
async def get_backtest_status(backtest_id: str):
    """Get the status of a backtest."""
    try:
        # Check if backtest exists
        if backtest_id not in running_backtests:
            raise HTTPException(status_code=404, detail=f"Backtest '{backtest_id}' not found")
        
        # Return status
        status = running_backtests[backtest_id]
        return BacktestStatusResponse(
            id=status["id"],
            strategy_name=status["strategy_name"],
            status=status["status"],
            progress=status["progress"],
            start_time=status["start_time"],
            end_time=status["end_time"],
            result_id=status["result_id"],
            error=status["error"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting backtest status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/results", response_model=List[Dict[str, Any]])
async def get_backtest_results():
    """Get all backtest results."""
    try:
        # Get results from results directory
        results_dir = ROOT_DIR / "results"
        if not results_dir.exists():
            return []
        
        # Load result files
        results = []
        for result_file in results_dir.glob("*.json"):
            with open(result_file, "r") as f:
                result = json.load(f)
                
                # Add summary info
                results.append({
                    "id": result["id"],
                    "strategy_name": result["config"]["strategy_name"],
                    "start_date": result["config"]["start_date"],
                    "end_date": result["config"]["end_date"],
                    "initial_capital": result["config"]["initial_capital"],
                    "symbols": result["config"]["symbols"],
                    "metrics": result["metrics"]
                })
        
        # Sort by date (newest first)
        results.sort(key=lambda x: x["end_date"], reverse=True)
        
        return results
    except Exception as e:
        logger.error(f"Error getting backtest results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/results/{result_id}", response_model=Dict[str, Any])
async def get_backtest_result(result_id: str):
    """Get a backtest result by ID."""
    try:
        # Check if result file exists
        result_file = ROOT_DIR / "results" / f"{result_id}.json"
        if not result_file.exists():
            raise HTTPException(status_code=404, detail=f"Backtest result '{result_id}' not found")
        
        # Load result
        with open(result_file, "r") as f:
            result = json.load(f)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting backtest result '{result_id}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/backtest/results/{result_id}", response_model=Dict[str, str])
async def delete_backtest_result(result_id: str):
    """Delete a backtest result."""
    try:
        # Check if result file exists
        result_file = ROOT_DIR / "results" / f"{result_id}.json"
        if not result_file.exists():
            raise HTTPException(status_code=404, detail=f"Backtest result '{result_id}' not found")
        
        # Delete result file
        result_file.unlink()
        
        logger.info(f"Deleted backtest result '{result_id}'")
        return {"message": f"Backtest result '{result_id}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting backtest result '{result_id}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/symbols", response_model=List[Dict[str, Any]])
async def get_symbols(query: Optional[str] = None):
    """Search for symbols."""
    try:
        # Get data manager
        data_manager = get_data_manager()
        
        # Search for symbols
        if query:
            assets = data_manager.search_symbols(query)
        else:
            # Return some default symbols if no query
            assets = [
                data_manager.get_asset_info("RELIANCE"),
                data_manager.get_asset_info("TCS"),
                data_manager.get_asset_info("INFY"),
                data_manager.get_asset_info("HDFCBANK"),
                data_manager.get_asset_info("ICICIBANK")
            ]
        
        # Convert to dictionaries
        symbols = []
        for asset in assets:
            symbols.append({
                "symbol": asset.symbol,
                "name": asset.name,
                "asset_type": asset.asset_type.value,
                "exchange": asset.exchange,
                "tradable": asset.tradable
            })
        
        return symbols
    except Exception as e:
        logger.error(f"Error searching symbols: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/data/{symbol}", response_model=Dict[str, Any])
async def get_market_data(
    symbol: str,
    interval: str = "1d",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100
):
    """Get historical market data for a symbol."""
    try:
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            # Calculate start date based on interval and limit
            if interval == "1m":
                start_date = end_date - timedelta(days=1)
            elif interval == "5m":
                start_date = end_date - timedelta(days=5)
            elif interval == "15m":
                start_date = end_date - timedelta(days=10)
            elif interval == "30m":
                start_date = end_date - timedelta(days=15)
            elif interval == "1h":
                start_date = end_date - timedelta(days=30)
            else:  # Daily or higher
                start_date = end_date - timedelta(days=limit)
        
        # Get data manager
        data_manager = get_data_manager()
        
        # Get historical data
        df = data_manager.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        if df.empty:
            return {
                "symbol": symbol,
                "interval": interval,
                "data": []
            }
        
        # Convert to list of dictionaries
        data = []
        for idx, row in df.iterrows():
            data.append({
                "timestamp": idx.isoformat() if isinstance(idx, pd.Timestamp) else idx,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"])
            })
        
        # Limit the number of data points
        if limit and len(data) > limit:
            data = data[-limit:]
        
        return {
            "symbol": symbol,
            "interval": interval,
            "data": data
        }
    except Exception as e:
        logger.error(f"Error getting market data for '{symbol}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def start_server():
    """Start the web server."""
    # Get app config
    app_config = Config.get_app_config()
    
    # Start server
    uvicorn.run(
        "kite.app.server:app",
        host=app_config["host"],
        port=app_config["port"],
        reload=app_config["debug"]
    )


if __name__ == "__main__":
    start_server()
