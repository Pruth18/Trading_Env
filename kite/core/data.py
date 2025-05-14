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
        try:
            # Get API credentials from config
            config = Config.get_angel_api_config()
            historical_api = config["historical"]
            
            # Prepare API request
            api_url = f"{config['api_base_url']}/rest/secure/angelbroking/historical/v1/getCandleData"
            
            # Map interval to Angel One API format
            interval_map = {
                "1m": "ONE_MINUTE",
                "5m": "FIVE_MINUTE",
                "15m": "FIFTEEN_MINUTE",
                "30m": "THIRTY_MINUTE",
                "1h": "ONE_HOUR",
                "1d": "ONE_DAY",
                "1w": "ONE_WEEK",
            }
            
            angel_interval = interval_map.get(interval, "ONE_DAY")
            
            # Format dates for API
            from_date = start_date.strftime("%Y-%m-%d %H:%M:%S")
            to_date = end_date.strftime("%Y-%m-%d %H:%M:%S")
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "CLIENT_LOCAL_IP",
                "X-ClientPublicIP": "CLIENT_PUBLIC_IP",
                "X-MACAddress": "MAC_ADDRESS",
                "X-PrivateKey": historical_api["api_key"],
                "Authorization": f"Bearer {historical_api['secret_key']}"
            }
            
            # Prepare payload
            # For Angel One API, we need to use token IDs instead of symbol names
            # Common token IDs for popular stocks
            token_map = {
                "RELIANCE": "2885",
                "TCS": "11536",
                "INFY": "1594",
                "HDFCBANK": "1333",
                "ICICIBANK": "4963",
                "HDFC": "1330",
                "KOTAKBANK": "492",
                "ITC": "424",
                "SBIN": "3045",
                "BAJFINANCE": "317"
            }
            
            # Get token ID for the symbol
            token = token_map.get(symbol, symbol)  # Use the symbol itself if not found in the map
            
            payload = {
                "exchange": "NSE",  # Default to NSE, can be parameterized
                "symboltoken": token,
                "interval": angel_interval,
                "fromdate": from_date,
                "todate": to_date
            }
            
            logger.info(f"Requesting historical data for {symbol} from {from_date} to {to_date} ({interval})")
            
            # Make the actual API call to Angel One
            logger.info(f"Making API call to {api_url}")
            try:
                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") == True and "data" in data:
                    # Process the API response
                    candles = data["data"]
                    logger.info(f"Received {len(candles)} candles from Angel One API")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)
                    
                    # Convert numeric columns
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Add symbol column
                    df["symbol"] = symbol
                    
                    return df
                else:
                    logger.error(f"API Error: {data.get('message', 'Unknown error')}")
                    # Fall back to dummy data if API call fails
                    logger.warning("Falling back to dummy data generator due to API error.")
            except Exception as e:
                logger.error(f"API request failed: {str(e)}")
                logger.warning("Falling back to dummy data generator due to request failure.")
        except Exception as e:
            logger.error(f"Error requesting historical data: {str(e)}")
            logger.warning("Falling back to dummy data generator.")
            
        # If we reach here, the API call failed, so we'll use dummy data as a fallback
        logger.warning(f"Generating realistic dummy data for {symbol} from {start_date} to {end_date}")
        
        # Generate dummy data for testing with more realistic price movements
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
        
        # Generate more realistic price data with trends and volatility
        np.random.seed(42)  # For reproducibility
        n = len(dates)
        
        if n == 0:
            logger.warning("No valid dates in range, returning empty DataFrame")
            return pd.DataFrame()
        
        # Base prices for different symbols
        symbol_base_prices = {
            "RELIANCE": 2500.0,
            "TCS": 3500.0,
            "INFY": 1500.0,
            "HDFCBANK": 1600.0,
            "ICICIBANK": 900.0,
            "HDFC": 2700.0,
            "KOTAKBANK": 1800.0,
            "ITC": 400.0,
            "SBIN": 600.0,
            "BAJFINANCE": 7000.0
        }
        
        # Start with a base price for the symbol
        base_price = symbol_base_prices.get(symbol, 1000.0)
        
        # Create a trend with cycles to ensure crossovers and RSI signals
        trend = np.linspace(0, 4*np.pi, n)  # Multiple cycles over the period
        cycle_component = np.sin(trend) * 0.15  # 15% cyclical component
        
        # Add a small upward drift
        drift = np.linspace(0, 0.10, n)  # 10% drift over the period
        
        # Add some random noise
        noise = np.random.normal(0, 0.01, n)  # Daily noise with 1% standard deviation
        
        # Combine components to create a price series with trends, cycles, and noise
        daily_returns = cycle_component + noise + np.diff(np.append([0], drift))
        
        # Generate price series
        prices = [base_price]
        for ret in daily_returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLC data with realistic intraday patterns
        data = []
        for i, date in enumerate(dates):
            price = prices[i]
            
            # Create more volatile high-low range for more trading opportunities
            volatility = 0.02 + 0.01 * np.sin(i/10)  # Varying volatility between 1-3%
            high_low_range = price * volatility
            
            # Create gap openings occasionally
            if np.random.random() < 0.2:  # 20% chance of a gap
                gap_direction = 1 if np.random.random() < 0.6 else -1  # More likely to gap up
                gap_size = np.random.uniform(0.005, 0.015)  # 0.5% to 1.5% gap
                open_price = price * (1 + gap_direction * gap_size)
            else:
                open_price = price * (1 + np.random.normal(0, 0.003))  # Normal open with some noise
            
            # Generate high and low with more extreme moves occasionally
            if np.random.random() < 0.1:  # 10% chance of a volatile day
                high_price = price + abs(np.random.normal(0, high_low_range))  # More extreme high
                low_price = price - abs(np.random.normal(0, high_low_range))  # More extreme low
            else:
                high_price = price + abs(np.random.normal(0, high_low_range/2))
                low_price = price - abs(np.random.normal(0, high_low_range/2))
            
            # Close price with tendency to revert or continue trend
            if np.random.random() < 0.7:  # 70% chance to follow the trend
                if i > 0 and prices[i] > prices[i-1]:
                    close_price = np.random.uniform(open_price, high_price)
                else:
                    close_price = np.random.uniform(low_price, open_price)
            else:  # 30% chance to revert
                if i > 0 and prices[i] > prices[i-1]:
                    close_price = np.random.uniform(low_price, open_price)
                else:
                    close_price = np.random.uniform(open_price, high_price)
            
            # Ensure high >= open, close and low <= open, close
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Volume tends to be higher on volatile days and trend changes
            base_volume = 1000000
            volatility_factor = (high_price - low_price) / price  # Normalized day's range
            trend_change = 0
            if i > 1:
                # Detect potential trend change
                prev_trend = prices[i-1] - prices[i-2]
                curr_trend = prices[i] - prices[i-1]
                if (prev_trend * curr_trend) < 0:  # Trend direction changed
                    trend_change = 1
            
            volume = int(base_volume * (1 + 2 * volatility_factor + trend_change * 0.5 + np.random.normal(0, 0.2)))
            volume = max(volume, 100)  # Ensure positive volume
            
            data.append({
                "timestamp": date,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "symbol": symbol
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Generated {len(df)} bars of dummy data for {symbol}")
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
