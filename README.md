# KITE - Personal Trading Environment

A modular and scalable trading environment for backtesting and deploying trading strategies with Angel One Broking API integration.

## Features

- Connect to Angel One Broking API for historical and live market data
- Create and manage multiple trading strategies
- Backtest strategies with historical data using vectorized execution
- View performance metrics (PnL, Sharpe Ratio, Drawdown, Win Rate)
- Compare multiple strategies side-by-side
- Web-based dashboard for strategy management and result visualization
- Modular design for easy integration of ML models and custom signals

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure your Angel One API credentials in `.env` file
4. Run the web server: `python -m kite.app.server`
5. Access the dashboard at http://localhost:8000

## Project Structure

```
kite/
├── core/               # Core trading engine components
│   ├── broker.py       # Broker interface (Angel One API)
│   ├── data.py         # Data management
│   ├── engine.py       # Trading engine
│   └── models.py       # Data models
├── strategies/         # Trading strategy implementations
│   ├── base.py         # Base strategy class
│   └── examples/       # Example strategies
├── backtest/           # Backtesting engine
│   ├── engine.py       # Backtesting logic
│   ├── metrics.py      # Performance metrics calculation
│   └── optimizer.py    # Parameter optimization
├── analysis/           # Analysis and visualization
│   ├── metrics.py      # Performance metrics
│   ├── plots.py        # Visualization utilities
│   └── reports.py      # Report generation
├── app/                # Web application
│   ├── server.py       # FastAPI server
│   ├── routes/         # API routes
│   └── ui/             # Dashboard UI
├── db/                 # Database models and utilities
│   ├── models.py       # SQLAlchemy models
│   └── session.py      # Database session management
└── utils/              # Utility functions
    ├── config.py       # Configuration management
    ├── logging.py      # Logging utilities
    └── helpers.py      # Helper functions
```

## Usage

See the documentation in the `docs/` directory for detailed usage instructions.
