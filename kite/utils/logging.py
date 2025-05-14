"""
Logging utilities for the KITE trading system.
Provides structured logging with rotation and different log levels.
"""

import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from loguru import logger

from kite.utils.config import LOGS_DIR


def setup_logger(
    strategy_name: Optional[str] = None,
    backtest_id: Optional[str] = None,
    level: str = "INFO"
) -> None:
    """
    Configure the logger for the application or a specific strategy/backtest.
    
    Args:
        strategy_name: Name of the strategy (if applicable)
        backtest_id: ID of the backtest run (if applicable)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Remove any existing handlers
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )
    
    # Create log filename based on context
    log_filename = "kite"
    if strategy_name:
        log_filename += f"_{strategy_name}"
    if backtest_id:
        log_filename += f"_{backtest_id}"
    
    # Add file handler with rotation
    log_path = LOGS_DIR / f"{log_filename}_{datetime.now().strftime('%Y%m%d')}.log"
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",    # Rotate when file reaches 10 MB
        retention="30 days", # Keep logs for 30 days
        compression="zip"    # Compress rotated logs
    )
    
    logger.info(f"Logger initialized: {log_path}")


def get_strategy_logger(strategy_name: str) -> logger:
    """
    Get a logger configured for a specific strategy.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Configured logger instance
    """
    setup_logger(strategy_name=strategy_name)
    return logger


def get_backtest_logger(strategy_name: str, backtest_id: str) -> logger:
    """
    Get a logger configured for a specific backtest run.
    
    Args:
        strategy_name: Name of the strategy
        backtest_id: ID of the backtest run
        
    Returns:
        Configured logger instance
    """
    setup_logger(strategy_name=strategy_name, backtest_id=backtest_id)
    return logger


def log_trade(
    strategy_name: str, 
    action: str,  # 'BUY' or 'SELL'
    symbol: str,
    quantity: int,
    price: float,
    timestamp: datetime,
    trade_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a trade execution with structured data.
    
    Args:
        strategy_name: Name of the strategy
        action: Trade action ('BUY' or 'SELL')
        symbol: Trading symbol
        quantity: Number of shares/contracts
        price: Execution price
        timestamp: Trade timestamp
        trade_id: Optional trade ID
        metadata: Additional trade metadata
    """
    log_data = {
        "strategy": strategy_name,
        "action": action,
        "symbol": symbol,
        "quantity": quantity,
        "price": price,
        "timestamp": timestamp.isoformat(),
        "trade_id": trade_id,
        **(metadata or {})
    }
    
    logger.info(f"TRADE: {log_data}")


def log_backtest_result(strategy_name: str, backtest_id: str, metrics: Dict[str, Any]) -> None:
    """
    Log backtest results with performance metrics.
    
    Args:
        strategy_name: Name of the strategy
        backtest_id: ID of the backtest run
        metrics: Dictionary of performance metrics
    """
    logger.info(f"BACKTEST RESULT: strategy={strategy_name}, id={backtest_id}, metrics={metrics}")


# Initialize default logger
setup_logger()
