"""
Test script to verify Angel One API connection and credentials.
"""

import os
import sys
import json
import pyotp
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from kite.utils.config import Config
from kite.utils.logging import setup_logger
from kite.core.broker import AngelOneBroker
from kite.core.data import DataManager
from loguru import logger

def test_broker_connection():
    """Test connection to Angel One Trading API."""
    logger.info("Testing connection to Angel One Trading API...")
    
    # Get broker instance
    broker = AngelOneBroker()
    
    # Print credentials (masked)
    logger.info(f"API Key: {broker.api_key[:4]}...{broker.api_key[-4:] if len(broker.api_key) > 8 else ''}")
    logger.info(f"Client ID: {broker.client_id[:2]}...{broker.client_id[-2:] if len(broker.client_id) > 4 else ''}")
    logger.info(f"Has Password: {'Yes' if broker.password else 'No'}")
    logger.info(f"Has TOTP Key: {'Yes' if broker.totp_key else 'No'}")
    
    # Test authentication
    auth_result = broker.authenticate()
    
    if auth_result:
        logger.success("Successfully authenticated with Angel One Trading API")
        logger.info(f"Session Token: {broker.session_token[:10]}...")
        logger.info(f"User Profile: {broker.user_profile}")
        return True
    else:
        logger.error("Failed to authenticate with Angel One Trading API")
        return False

def test_historical_data():
    """Test fetching historical data from Angel One API."""
    logger.info("Testing historical data retrieval...")
    
    # Get data manager instance
    data_manager = DataManager()
    
    # Test symbols
    symbols = ["RELIANCE", "TCS", "INFY"]
    
    # Test intervals
    intervals = ["1d", "1h", "15m"]
    
    # Test period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    for symbol in symbols:
        for interval in intervals:
            logger.info(f"Fetching {interval} data for {symbol} from {start_date.date()} to {end_date.date()}")
            
            try:
                df = data_manager.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    use_cache=False,
                    force_download=True
                )
                
                if df is not None and not df.empty:
                    logger.success(f"Successfully retrieved {len(df)} bars for {symbol} ({interval})")
                    logger.info(f"Sample data: \n{df.head(3)}")
                else:
                    logger.warning(f"No data retrieved for {symbol} ({interval})")
            except Exception as e:
                logger.error(f"Error retrieving data for {symbol} ({interval}): {str(e)}")
    
    return True

def test_market_data():
    """Test fetching market data from Angel One API."""
    logger.info("Testing market data retrieval...")
    
    # Get data manager instance
    data_manager = DataManager()
    
    # Test symbols
    symbols = ["RELIANCE", "TCS", "INFY"]
    
    for symbol in symbols:
        logger.info(f"Fetching latest price for {symbol}")
        
        try:
            price = data_manager.get_latest_price(symbol)
            logger.success(f"Latest price for {symbol}: {price}")
        except Exception as e:
            logger.error(f"Error retrieving latest price for {symbol}: {str(e)}")
    
    return True

def test_symbol_search():
    """Test symbol search functionality."""
    logger.info("Testing symbol search...")
    
    # Get data manager instance
    data_manager = DataManager()
    
    # Test search queries
    queries = ["REL", "BANK", "INFO"]
    
    for query in queries:
        logger.info(f"Searching for symbols matching '{query}'")
        
        try:
            results = data_manager.search_symbols(query)
            logger.success(f"Found {len(results)} symbols matching '{query}'")
            for asset in results:
                logger.info(f"  {asset.symbol} - {asset.name} ({asset.exchange})")
        except Exception as e:
            logger.error(f"Error searching for symbols matching '{query}': {str(e)}")
    
    return True

def main():
    """Run all tests."""
    # Setup logger
    setup_logger(level="INFO")
    
    logger.info("Starting Angel One API connection tests...")
    
    # Print configuration
    config = Config.get_angel_api_config()
    logger.info(f"API Base URL: {config['api_base_url']}")
    logger.info(f"Trading API App: {config['trading']['app_name']}")
    logger.info(f"Historical API App: {config['historical']['app_name']}")
    logger.info(f"Market API App: {config['market']['app_name']}")
    
    # Run tests
    broker_test = test_broker_connection()
    historical_test = test_historical_data()
    market_test = test_market_data()
    symbol_test = test_symbol_search()
    
    # Print results
    logger.info("\n--- Test Results ---")
    logger.info(f"Broker Connection: {'✅ PASS' if broker_test else '❌ FAIL'}")
    logger.info(f"Historical Data: {'✅ PASS' if historical_test else '❌ FAIL'}")
    logger.info(f"Market Data: {'✅ PASS' if market_test else '❌ FAIL'}")
    logger.info(f"Symbol Search: {'✅ PASS' if symbol_test else '❌ FAIL'}")
    
    if all([broker_test, historical_test, market_test, symbol_test]):
        logger.success("All tests passed! Your KITE trading system is ready to use.")
    else:
        logger.warning("Some tests failed. Please check the logs for details.")

if __name__ == "__main__":
    main()
