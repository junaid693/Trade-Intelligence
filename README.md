# Trade Intelligence

Trade Intelligence is a local market-data ingestion and intelligence system focused exclusively on Binance market data.

## Project Scope & Constraints

- **Exchange**: Binance only
- **Data Access**: Binance official public market-data APIs
- **Database**: SQLite (planned for subsequent phases; not implemented yet)
- **Deployment**: Local initially
- **Trading**: Completely disabled (no execution, order routing, or portfolio management)
- **API Credentials**: Not required (read-only public market data only)
- **Out of Scope**: No AI/ML, trading recommendations, automated trading, technical analysis engines, Binance Square integrations, or frontend in this phase.

## Current Phase: Phase 1.1

Phase 1.1 focuses strictly on initializing the backend development environment:
- Establishing a clean Python package layout (`src/trade_intelligence/`)
- Setting up a local Python virtual environment (`.venv`)
- Defining minimal dependencies required for public REST API market-data connectivity
- Establishing repository `.gitignore` and workspace documentation

## Development Setup

### 1. Prerequisites
- Python 3.10+ (tested with Python 3.14)

### 2. Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

## Project Structure
```text
Trade-Intelligence/
├── .gitignore          # Git exclusion rules
├── README.md           # Project documentation and Phase 1.1 scope
├── requirements.txt    # Minimal dependencies for Binance public API
├── trade_intelligence/
│   └── __init__.py     # Core package root
└── tests/
    └── __init__.py     # Test suite root
```