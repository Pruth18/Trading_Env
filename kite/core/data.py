"""
Data management module for the KITE trading system.
Handles fetching, processing, and storing market data.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
import requests
from pathlib import Path

from loguru import logger

from kite.core.models import Bar, Asset, AssetType
from kite.utils.config import Config, DATA_DIR


class DataManager:
    """
    Data manager for fetching, processing, and storing market data.
    Handles interactions with data sources and provides a unified interface
    for accessing market data.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the data manager.
        
        Args:
            cache_dir: Directory for caching data (defaults to DATA_DIR)
        """
        self.cache_dir = cache_dir or DATA_DIR
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for different data types
        self.historical_dir = self.cache_dir / "historical"
        self.historical_dir.mkdir(exist_ok=True)
        
        self.intraday_dir = self.cache_dir / "intraday"
        self.intraday_dir.mkdir(exist_ok=True)
        
        self.metadata_dir = self.cache_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Data caches
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._asset_cache: Dict[str, Asset] = {}
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        use_cache: bool = True,
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Get historical price data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval (e.g., "1m", "5m", "1h", "1d")
            use_cache: Whether to use cached data if available
            force_download: Whether to force download even if cached data exists
            
        Returns:
            DataFrame with historical price data (OHLCV)
        """
        cache_key = f"{symbol}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        # Check if data is in memory cache
        if use_cache and cache_key in self._data_cache and not force_download:
            logger.debug(f"Using memory-cached data for {cache_key}")
            return self._data_cache[cache_key]
        
        # Check if data is in file cache
        cache_file = self.historical_dir / f"{cache_key}.csv"
        if use_cache and cache_file.exists() and not force_download:
            logger.debug(f"Loading cached data from {cache_file}")
            df = pd.read_csv(cache_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            self._data_cache[cache_key] = df
            return df
        
        # Download data from API
        logger.info(f"Downloading historical data for {symbol} from {start_date} to {end_date} ({interval})")
        df = self._download_historical_data(symbol, start_date, end_date, interval)
        
        # Cache data
        if use_cache and not df.empty:
            # Save to file cache
            df.to_csv(cache_file)
            # Save to memory cache
            self._data_cache[cache_key] = df
        
        return df
    
    def _download_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Download historical data from the Angel One API.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval (e.g., "1m", "5m", "1h", "1d")
            
        Returns:
            DataFrame with historical price data (OHLCV)
        """
        # This is a placeholder for the actual API implementation
        # In a real implementation, you would use the Angel One API client
        
        # For now, we'll generate some dummy data for testing
        logger.warning("Using dummy data generator. Replace with actual Angel One API implementation.")
        
        # Map interval to timedelta
        interval_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1),
            "1w": timedelta(weeks=1),
        }
        
        delta = interval_map.get(interval, timedelta(days=1))
        
        # Generate date range
        dates = []
        current_date = start_date
        while current_date <= end_date:
            if interval.endswith('d') or interval.endswith('w'):
                # Skip weekends for daily and weekly data
                if current_date.weekday() < 5:  # 0-4 are Monday to Friday
                    dates.append(current_date)
            else:
                # For intraday data, only include times during market hours (9:15 AM to 3:30 PM)
                if 9 <= current_date.hour < 15 or (current_date.hour == 15 and current_date.minute <= 30):
                    dates.append(current_date)
            
            current_date += delta
        
        if not dates:
            return pd.DataFrame()
        
        # Generate dummy price data
        np.random.seed(42)  # For reproducibility
        
        # Start with a base price
        base_price = 1000.0
        
        # Generate price data with some randomness but following a general trend
        data = []
        prev_close = base_price
        
        for date in dates:
            # Random daily return between -2% and 2%
            daily_return = np.random.normal(0.0001, 0.015)
            
            # Calculate OHLC based on previous close and daily return
            close = prev_close * (1 + daily_return)
            high = close * (1 + abs(np.random.normal(0, 0.005)))
            low = close * (1 - abs(np.random.normal(0, 0.005)))
            open_price = prev_close * (1 + np.random.normal(0, 0.003))
            
            # Ensure high is the highest and low is the lowest
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Random volume
            volume = int(np.random.normal(1000000, 500000))
            if volume < 0:
                volume = 100000
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'symbol': symbol,
            })
            
            prev_close = close
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_intraday_data(
        self,
        symbol: str,
        interval: str = "5m",
        days_back: int = 1,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get intraday price data for a symbol.
        
        Args:
            symbol: Trading symbol
            interval: Data interval (e.g., "1m", "5m", "15m", "30m", "1h")
            days_back: Number of days to look back
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with intraday price data (OHLCV)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        return self.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            use_cache=use_cache
        )
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest price
        """
        # In a real implementation, you would use the Angel One API to get the latest price
        # For now, we'll use the last price from the intraday data
        df = self.get_intraday_data(symbol, interval="5m", days_back=1)
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        return df['close'].iloc[-1]
    
    def get_asset_info(self, symbol: str, force_refresh: bool = False) -> Asset:
        """
        Get information about a tradable asset.
        
        Args:
            symbol: Trading symbol
            force_refresh: Whether to force refresh from the API
            
        Returns:
            Asset object with symbol information
        """
        if symbol in self._asset_cache and not force_refresh:
            return self._asset_cache[symbol]
        
        # Check if asset info is cached in file
        cache_file = self.metadata_dir / f"{symbol}_info.json"
        if cache_file.exists() and not force_refresh:
            with open(cache_file, "r") as f:
                asset_data = json.load(f)
                asset = Asset(
                    symbol=asset_data["symbol"],
                    name=asset_data["name"],
                    asset_type=AssetType(asset_data["asset_type"]),
                    exchange=asset_data["exchange"],
                    tradable=asset_data["tradable"],
                    tick_size=asset_data["tick_size"],
                    lot_size=asset_data["lot_size"],
                    metadata=asset_data.get("metadata", {})
                )
                self._asset_cache[symbol] = asset
                return asset
        
        # In a real implementation, you would use the Angel One API to get asset info
        # For now, we'll create a dummy asset
        logger.warning(f"Using dummy asset info for {symbol}. Replace with actual Angel One API implementation.")
        
        asset = Asset(
            symbol=symbol,
            name=f"{symbol} Stock",
            asset_type=AssetType.EQUITY,
            exchange="NSE",
            tradable=True,
            tick_size=0.05,
            lot_size=1,
            metadata={}
        )
        
        # Cache asset info
        self._asset_cache[symbol] = asset
        
        # Save to file cache
        with open(cache_file, "w") as f:
            json.dump({
                "symbol": asset.symbol,
                "name": asset.name,
                "asset_type": asset.asset_type.value,
                "exchange": asset.exchange,
                "tradable": asset.tradable,
                "tick_size": asset.tick_size,
                "lot_size": asset.lot_size,
                "metadata": asset.metadata
            }, f, indent=4)
        
        return asset
    
    def search_symbols(self, query: str) -> List[Asset]:
        """
        Search for symbols matching a query.
        
        Args:
            query: Search query
            
        Returns:
            List of matching assets
        """
        # In a real implementation, you would use the Angel One API to search for symbols
        # For now, we'll return some dummy results
        logger.warning(f"Using dummy symbol search for '{query}'. Replace with actual Angel One API implementation.")
        
        dummy_results = [
            Asset(
                symbol="RELIANCE",
                name="Reliance Industries Ltd",
                asset_type=AssetType.EQUITY,
                exchange="NSE",
                tradable=True
            ),
            Asset(
                symbol="TCS",
                name="Tata Consultancy Services Ltd",
                asset_type=AssetType.EQUITY,
                exchange="NSE",
                tradable=True
            ),
            Asset(
                symbol="INFY",
                name="Infosys Ltd",
                asset_type=AssetType.EQUITY,
                exchange="NSE",
                tradable=True
            ),
            Asset(
                symbol="HDFCBANK",
                name="HDFC Bank Ltd",
                asset_type=AssetType.EQUITY,
                exchange="NSE",
                tradable=True
            ),
            Asset(
                symbol="ICICIBANK",
                name="ICICI Bank Ltd",
                asset_type=AssetType.EQUITY,
                exchange="NSE",
                tradable=True
            )
        ]
        
        # Filter by query
        query = query.lower()
        results = [asset for asset in dummy_results if query in asset.symbol.lower() or query in asset.name.lower()]
        
        return results
    
    def get_market_hours(self, exchange: str = "NSE") -> Tuple[datetime, datetime]:
        """
        Get market hours for an exchange.
        
        Args:
            exchange: Exchange code
            
        Returns:
            Tuple of (market_open, market_close) datetimes
        """
        # In a real implementation, you would use the Angel One API to get market hours
        # For now, we'll return fixed hours for NSE
        
        now = datetime.now()
        market_open = datetime(now.year, now.month, now.day, 9, 15)  # 9:15 AM
        market_close = datetime(now.year, now.month, now.day, 15, 30)  # 3:30 PM
        
        return market_open, market_close
    
    def is_market_open(self, exchange: str = "NSE") -> bool:
        """
        Check if the market is currently open.
        
        Args:
            exchange: Exchange code
            
        Returns:
            True if market is open, False otherwise
        """
        now = datetime.now()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False
        
        market_open, market_close = self.get_market_hours(exchange)
        
        return market_open <= now <= market_close
    
    def clear_cache(self, symbol: Optional[str] = None, interval: Optional[str] = None):
        """
        Clear data cache.
        
        Args:
            symbol: If provided, clear cache only for this symbol
            interval: If provided, clear cache only for this interval
        """
        # Clear memory cache
        if symbol is None and interval is None:
            self._data_cache.clear()
            self._asset_cache.clear()
        else:
            keys_to_remove = []
            for key in self._data_cache:
                if (symbol is None or symbol in key) and (interval is None or interval in key):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._data_cache[key]
            
            if symbol is not None:
                if symbol in self._asset_cache:
                    del self._asset_cache[symbol]
        
        # Clear file cache
        if symbol is None and interval is None:
            # Clear all cache files
            for file in self.historical_dir.glob("*.csv"):
                file.unlink()
            for file in self.intraday_dir.glob("*.csv"):
                file.unlink()
            for file in self.metadata_dir.glob("*.json"):
                file.unlink()
        else:
            # Clear specific cache files
            for file in self.historical_dir.glob("*.csv"):
                if (symbol is None or symbol in file.name) and (interval is None or interval in file.name):
                    file.unlink()
            
            for file in self.intraday_dir.glob("*.csv"):
                if (symbol is None or symbol in file.name) and (interval is None or interval in file.name):
                    file.unlink()
            
            if symbol is not None:
                metadata_file = self.metadata_dir / f"{symbol}_info.json"
                if metadata_file.exists():
                    metadata_file.unlink()


# Singleton instance
_data_manager = None


def get_data_manager() -> DataManager:
    """
    Get the singleton instance of the DataManager.
    
    Returns:
        DataManager instance
    """
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    
    return _data_manager
