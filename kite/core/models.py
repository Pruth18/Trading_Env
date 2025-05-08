"""
Core data models for the KITE trading system.
Defines the data structures used throughout the application.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import uuid
import pandas as pd
import numpy as np


class OrderType(str, Enum):
    """Order types supported by the system."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(str, Enum):
    """Order sides (buy/sell)."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Order statuses."""
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(str, Enum):
    """Time in force options."""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


class AssetType(str, Enum):
    """Asset types supported by the system."""
    EQUITY = "EQUITY"
    FUTURE = "FUTURE"
    OPTION = "OPTION"
    CURRENCY = "CURRENCY"
    COMMODITY = "COMMODITY"


class PositionSide(str, Enum):
    """Position sides."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class StrategyType(str, Enum):
    """Types of trading strategies."""
    TREND_FOLLOWING = "TREND_FOLLOWING"
    MEAN_REVERSION = "MEAN_REVERSION"
    BREAKOUT = "BREAKOUT"
    MOMENTUM = "MOMENTUM"
    ARBITRAGE = "ARBITRAGE"
    STATISTICAL = "STATISTICAL"
    MACHINE_LEARNING = "MACHINE_LEARNING"
    CUSTOM = "CUSTOM"


@dataclass
class Asset:
    """Represents a tradable asset."""
    symbol: str
    name: str
    asset_type: AssetType
    exchange: str
    tradable: bool = True
    tick_size: float = 0.05
    lot_size: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Bar:
    """Represents a price bar (OHLCV)."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str
    interval: str  # e.g., "1m", "5m", "1h", "1d"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Bar':
        """Create a Bar from a dictionary."""
        return cls(
            timestamp=data['timestamp'] if isinstance(data['timestamp'], datetime) else pd.to_datetime(data['timestamp']),
            open=float(data['open']),
            high=float(data['high']),
            low=float(data['low']),
            close=float(data['close']),
            volume=int(data['volume']),
            symbol=data['symbol'],
            interval=data['interval']
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'symbol': self.symbol,
            'interval': self.interval
        }


@dataclass
class Order:
    """Represents a trading order."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.CREATED
    filled_quantity: int = 0
    average_fill_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    rejected_reason: Optional[str] = None
    strategy_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'average_fill_price': self.average_fill_price,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'cancelled_at': self.cancelled_at.isoformat() if self.cancelled_at else None,
            'rejected_reason': self.rejected_reason,
            'strategy_id': self.strategy_id,
            'broker_order_id': self.broker_order_id
        }


@dataclass
class Trade:
    """Represents a trade execution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    commission: float = 0.0
    strategy_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'commission': self.commission,
            'strategy_id': self.strategy_id
        }


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: int = 0
    average_price: float = 0.0
    side: PositionSide = PositionSide.FLAT
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    open_trades: List[Trade] = field(default_factory=list)
    closed_trades: List[Trade] = field(default_factory=list)
    
    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.side != PositionSide.FLAT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'average_price': self.average_price,
            'side': self.side.value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'open_trades': [trade.to_dict() for trade in self.open_trades],
            'closed_trades': [trade.to_dict() for trade in self.closed_trades]
        }


@dataclass
class Portfolio:
    """Represents a trading portfolio."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Default Portfolio"
    initial_capital: float = 100000.0
    cash: float = 100000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    
    @property
    def equity(self) -> float:
        """Calculate total equity (cash + position values)."""
        position_value = sum(pos.quantity * pos.average_price for pos in self.positions.values())
        return self.cash + position_value
    
    @property
    def realized_pnl(self) -> float:
        """Calculate total realized PnL."""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate total unrealized PnL."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'equity': self.equity,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()}
        }


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str
    type: StrategyType = StrategyType.CUSTOM
    description: str = ""
    symbols: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.type.value,
            'description': self.description,
            'symbols': self.symbols,
            'parameters': self.parameters,
            'enabled': self.enabled
        }


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_name: str = ""
    start_date: datetime = field(default_factory=lambda: datetime.now().replace(year=datetime.now().year-1))
    end_date: datetime = field(default_factory=datetime.now)
    initial_capital: float = 100000.0
    symbols: List[str] = field(default_factory=list)
    interval: str = "1d"  # e.g., "1m", "5m", "1h", "1d"
    commission_rate: float = 0.0020  # 0.20%
    slippage_rate: float = 0.0005  # 0.05%
    strategy_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'strategy_name': self.strategy_name,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'symbols': self.symbols,
            'interval': self.interval,
            'commission_rate': self.commission_rate,
            'slippage_rate': self.slippage_rate,
            'strategy_parameters': self.strategy_parameters
        }


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    id: str
    config: BacktestConfig
    equity_curve: pd.DataFrame  # Columns: timestamp, equity, cash, holdings
    trades: List[Trade]
    metrics: Dict[str, float]
    positions: Dict[str, List[Position]]
    logs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'config': self.config.to_dict(),
            'equity_curve': self.equity_curve.to_dict(orient='records'),
            'trades': [trade.to_dict() for trade in self.trades],
            'metrics': self.metrics,
            'positions': {symbol: [pos.to_dict() for pos in pos_list] for symbol, pos_list in self.positions.items()},
            'logs': self.logs
        }
