# LEAP Strategic Asset Engine (LSAE)

An AI-powered Python system for discovering and analyzing LEAP options opportunities using a multi-agent architecture.

## 🎯 Overview

The LEAP Strategic Asset Engine automates the research and analysis of Long-term Equity AnticiPation Securities (LEAP options) using a **"Buy 1.5yr, Sell 1yr"** strategy. It combines AI-driven market discovery with rigorous quantitative analysis and risk assessment.

### Key Features

- **🤖 AI-Powered Discovery**: Uses Grok to scan news and social media for trending investment themes
- **📊 Automated Options Analysis**: Fetches real-time option chains and Greeks via yfinance
- **🧮 Chain-of-Thought Reasoning**: DeepSeek R1 performs deep quantitative analysis with **real-time streaming output**
- **🛡️ Risk Assessment**: DeepSeek V3 acts as "devil's advocate" to identify red flags
- **💾 Audit Trail**: SQLite database tracks all recommendations and AI reasoning
- **📄 Trade Cards**: Generates detailed markdown reports with **complete analysis** (no truncation)

## 🏗️ Architecture

The system features **intelligent thought buffering** for coherent AI reasoning:

```
┌─────────────────────────────────────────────────────────┐
│                   Orchestrator                          │
│            (Manages workflow & state)                   │
└─────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Discovery  │ │   Market    │ │    Quant    │ │    Risk     │
│    Scout    │ │    Data     │ │  Reasoning  │ │   Critic    │
│   (Grok)    │ │  (yfinance) │ │(DeepSeek R1)│ │(DeepSeek V3)│
│             │ │             │ │  Streaming  │ │             │
│             │ │             │ │   Output    │ │             │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
         │              │              │              │
         └──────────────┴──────────────┴──────────────┴──────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   Database    │
                    │   (SQLite)    │
                    └───────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- OpenRouter API key ([Get one here](https://openrouter.ai))

### Virtual Environment Setup (Recommended)

1. **Create a virtual environment**
```bash
python -m venv leap_env
```

2. **Activate the virtual environment**

**On macOS/Linux:**
```bash
source leap_env/bin/activate
```

3. **Verify activation** (your prompt should show `(leap_env)`)

### Setup

1. **Clone or download the project**
```bash
cd leap_engine
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

4. **Create required directories**
```bash
mkdir -p database logs outputs
```

## 🚀 Usage

### Mode 1: Full Pipeline (Discovery + Analysis)

Discover trending assets and analyze them automatically:

```bash
python main.py
```

**Options:**
- `--sectors 5`: Number of sectors to discover (default: 5)
- `--tickers-per-sector 2`: Tickers per sector (default: 2)

**Example:**
```bash
python main.py --sectors 3 --tickers-per-sector 3
```

### Mode 2: Manual Single Ticker Analysis

Analyze a specific ticker with a custom thesis:

```bash
python main.py --ticker NVDA --narrative "AI infrastructure leader benefiting from GPU demand"
```

### Mode 3: Export Approved Trades

Export all approved recommendations to markdown:

```bash
python main.py --export
```

## 🧮 LEAP Strategy Parameters

The engine evaluates options based on these criteria:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Time to Expiry** | 540+ days (1.5 years) | Ensures sufficient time value |
| **Target Delta** | 0.80 | Deep ITM acts as stock substitute |
| **Exit Point** | Day 365 (1 year) | Leaves 6 months of extrinsic value |
| **IV Percentile** | < 40th percentile | Avoid buying when volatility is elevated |
| **Bid/Ask Spread** | < 5% | Ensures liquidity |

## 📊 Output Examples

### Console Output
```
🎯 Starting LEAP Strategic Asset Engine Pipeline
==============================================================

📡 Phase 1: Discovery Scout - Scanning for trending assets...
✅ Discovered 10 assets

==============================================================
📊 Analyzing 1/10: NVDA
==============================================================
  📈 Fetching market data for NVDA...
  ✅ Found 8 LEAP candidates
  🧮 Running quantitative analysis...
🤔 First, I need to calculate the break-even price for this LEAP trade.
🤔 The strike price is $160 with a premium of $58.85, so break-even equals...
🤔 Performing Black-Scholes calculations for scenario analysis...

  ✅ Completed quant analysis for NVDA
  🛡️ Running risk assessment...
  ✅ Risk critique completed for NVDA

  ✅ APPROVED
  Contract: NVDA $120 Call 2026-06-19 (500 days)
  Entry: $45.50 | Projected 1Y: $58.20 (+28.0%)
  Risk Score: 0.25/1.0
```

### Trade Card (Markdown)

See [example trade card](docs/example_trade_card.md) for a full markdown export.

## 🗄️ Database Schema

The engine maintains a SQLite database with these tables:

- **`trending_assets`**: Discovered assets from the scout
- **`trade_cards`**: Complete trade recommendations
- **`trade_performance`**: Track actual vs. projected returns (future)
- **`reasoning_log`**: Full AI reasoning audit trail

Query examples:
```python
from database.db import Database
from config import config

db = Database(config.database.path)

# Get recent approved trades
approved = db.get_recent_trade_cards(limit=10, approved_only=True)

# Get all recommendations for NVDA
history = db.get_ticker_history("NVDA")
```

## 🔧 Configuration

Edit `.env` to customize:

```bash
# Model Selection
GROK_MODEL=x-ai/grok-beta
DEEPSEEK_R1_MODEL=deepseek/deepseek-r1
DEEPSEEK_V3_MODEL=deepseek/deepseek-chat

# Trading Parameters
MIN_DAYS_TO_EXPIRY=540
TARGET_DELTA=0.80
EXIT_DAY=365
MAX_IV_PERCENTILE=40
MAX_BID_ASK_SPREAD_PCT=5.0
```

## � Security & Data Protection

The project includes comprehensive security measures:

- **API Key Protection**: `.env` files are automatically ignored by `.gitignore`
- **Virtual Environment Isolation**: `leap_env/` directory never committed
- **Generated Content**: `outputs/`, `logs/`, and `database/` directories protected
- **No Sensitive Data**: Only template files (`.env.example`) are tracked

**Before committing to GitHub:**
```bash
git status --ignored  # Verify sensitive files are ignored
```

## �� Project Structure

```
leap_engine/
├── main.py                 # Entry point
├── orchestrator.py         # Main workflow controller
├── config.py              # Configuration management
├── requirements.txt       # Dependencies
├── .env.example          # Environment template
├── agents/
│   ├── __init__.py
│   └── ai_agents.py      # Discovery, Reasoning, Critique
├── data/
│   ├── __init__.py
│   └── market_data_client.py  # yfinance integration
├── database/
│   ├── __init__.py
│   └── db.py             # SQLite persistence
├── models/
│   └── __init__.py       # Pydantic data models
├── database/
│   └── leap_engine.db    # SQLite database (created on first run)
├── logs/
│   └── leap_engine.log   # Application logs
└── outputs/
    └── *_trade_card.md   # Exported trade cards
```

## 🔬 Under the Hood

### 1. Discovery Scout (Grok)
- Scans financial news and X/Twitter for trending narratives
- Returns tickers with sector context and conviction scores
- Focuses on liquid, institutional-quality names

### 2. Market Data Client (yfinance)
- Fetches real-time spot prices
- Downloads full option chains (all expirations)
- Filters for LEAP criteria (540+ days, 0.70-0.90 delta)
- Calculates IV rank and 52-week ranges

### 3. Quant Reasoning Engine (DeepSeek R1)
- Performs 3-point scenario analysis (Bull +20%, Base +5%, Bear -15%)
- Calculates break-even prices
- Estimates option value at 1-year exit point
- Uses Black-Scholes logic with time decay
- Provides conviction score

### 4. Risk Critic (DeepSeek V3)
- Checks for elevated IV (> 40th percentile)
- Flags wide bid/ask spreads
- Identifies upcoming earnings in hold period
- Assesses liquidity risks
- Final approve/reject decision

## ⚠️ Important Notes

### Limitations
- **Not Financial Advice**: This is a research tool, not investment advice
- **Data Delays**: yfinance data may have slight delays
- **IV Calculation**: Historical IV rank is simplified (placeholder)
- **Greeks Accuracy**: Depends on yfinance data quality
- **API Costs**: OpenRouter charges per token (DeepSeek is cheap, Grok varies)

### Best Practices
1. **Always verify** option quotes in your broker before trading
2. **Check earnings dates** manually - they can change
3. **Monitor IV rank** - the system uses a simplified calculation
4. **Paper trade first** - validate the strategy before using real capital
5. **Review critique reasoning** - don't blindly trust approvals

## 🛠️ Extending the System

### Add Custom Risk Checks
Edit `agents/ai_agents.py` → `RiskCritic._build_critique_prompt()`

### Change LEAP Criteria
Edit `.env` parameters or `config.py` defaults

### Add New Data Sources
Extend `data/market_data_client.py` with alternative APIs

### Custom Output Formats
Modify `TradeCard.to_markdown()` in `models/__init__.py`

## 📚 Resources

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [LEAP Options Strategy Guide](https://www.investopedia.com/terms/l/leaps.asp)

## 🤝 Contributing

This is a research project. Feel free to:
- Fork and modify for your own use
- Submit issues for bugs
- Share improvements via pull requests

## 📄 License

MIT License - see LICENSE file for details

## ⚡ Quick Start Example

```bash
# 1. Setup
pip install -r requirements.txt
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env

# 2. Test with a single ticker
python main.py --ticker AAPL --narrative "iPhone supercycle thesis"

# 3. Run full discovery
python main.py --sectors 3 --tickers-per-sector 2

# 4. Export results
python main.py --export
```

## 🎓 Learning Path

1. **Start Simple**: Analyze 1-2 manual tickers to understand the output
2. **Read the Critique**: The Risk Critic's reasoning is educational
3. **Check Historical Accuracy**: Compare projections vs. actual outcomes
4. **Tune Parameters**: Adjust LEAP criteria based on your risk tolerance
5. **Backtest**: Track performance over time using the database

---

**Built with**: Python 🐍 | OpenRouter 🤖 | yfinance 📊 | SQLite 💾

*Remember: This tool is for research and education. Always do your own due diligence before investing.*
