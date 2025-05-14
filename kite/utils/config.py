"""
Configuration management for the KITE trading system.
Handles loading environment variables and configuration settings.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
CONFIG_DIR = ROOT_DIR / "config"

# Ensure directories exist
for directory in [DATA_DIR, LOGS_DIR, CONFIG_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Angel One API configuration
ANGEL_API_CONFIG = {
    # Common credentials
    "client_id": os.getenv("ANGEL_CLIENT_ID", "YOUR_CLIENT_ID"),  # Replace with your actual client ID
    "password": os.getenv("ANGEL_PASSWORD", "YOUR_PASSWORD"),  # Replace with your actual password
    "totp_key": os.getenv("ANGEL_TOTP_KEY", "YOUR_TOTP_KEY"),  # Replace with your actual TOTP key
    "api_base_url": os.getenv("ANGEL_API_URL", "https://apiconnect.angelbroking.com"),
    
    # Trading API credentials
    "trading": {
        "api_key": os.getenv("ANGEL_TRADING_API_KEY", "ZNQY5zne"),
        "secret_key": os.getenv("ANGEL_TRADING_SECRET_KEY", "aad2fa65-641a-42e2-84d7-842aafbf6f64"),
        "app_name": "MyTradingApi"
    },
    
    # Historical Data API credentials
    "historical": {
        "api_key": os.getenv("ANGEL_HISTORICAL_API_KEY", "10XN79Ba"),
        "secret_key": os.getenv("ANGEL_HISTORICAL_SECRET_KEY", "28abda57-e57f-4340-8809-f6124d2d8a53"),
        "app_name": "MyHistoricalApi"
    },
    
    # Market Feeds API credentials
    "market": {
        "api_key": os.getenv("ANGEL_MARKET_API_KEY", "nf3HXMX1"),
        "secret_key": os.getenv("ANGEL_MARKET_SECRET_KEY", "ca906d28-bd5f-452c-acb8-2177b3ef3d72"),
        "app_name": "MyMarketApi"
    }
}

# Database configuration
DB_CONFIG = {
    "uri": os.getenv("DB_URI", f"sqlite:///{ROOT_DIR}/kite.db"),
    "echo": os.getenv("DB_ECHO", "False").lower() == "true",
}

# Web app configuration
APP_CONFIG = {
    "host": os.getenv("APP_HOST", "127.0.0.1"),
    "port": int(os.getenv("APP_PORT", "8000")),
    "debug": os.getenv("APP_DEBUG", "False").lower() == "true",
}

# Default backtest configuration
DEFAULT_BACKTEST_CONFIG = {
    "initial_capital": 100000,
    "commission": 0.0020,  # 0.20%
    "slippage": 0.0005,    # 0.05%
}


class Config:
    """Configuration manager for the KITE trading system."""
    
    @staticmethod
    def get_angel_api_config() -> Dict[str, str]:
        """Get Angel One API configuration."""
        return ANGEL_API_CONFIG
    
    @staticmethod
    def get_db_config() -> Dict[str, Any]:
        """Get database configuration."""
        return DB_CONFIG
    
    @staticmethod
    def get_app_config() -> Dict[str, Any]:
        """Get web app configuration."""
        return APP_CONFIG
    
    @staticmethod
    def get_backtest_config() -> Dict[str, Any]:
        """Get default backtest configuration."""
        return DEFAULT_BACKTEST_CONFIG
    
    @staticmethod
    def save_strategy_config(strategy_name: str, config: Dict[str, Any]) -> None:
        """Save strategy configuration to file."""
        config_file = CONFIG_DIR / f"{strategy_name}.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
    
    @staticmethod
    def load_strategy_config(strategy_name: str) -> Optional[Dict[str, Any]]:
        """Load strategy configuration from file."""
        config_file = CONFIG_DIR / f"{strategy_name}.json"
        if not config_file.exists():
            return None
        
        with open(config_file, "r") as f:
            return json.load(f)
    
    @staticmethod
    def list_strategy_configs() -> Dict[str, Dict[str, Any]]:
        """List all available strategy configurations."""
        configs = {}
        for config_file in CONFIG_DIR.glob("*.json"):
            strategy_name = config_file.stem
            with open(config_file, "r") as f:
                configs[strategy_name] = json.load(f)
        return configs
