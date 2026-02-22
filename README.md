# LEAP Strategic Asset Engine (LSAE)

An AI-powered Python system for discovering, analyzing, and backtesting LEAP options strategies using a multi-agent architecture with temporal constraints.

## 🎯 Overview

The LEAP Strategic Asset Engine is a comprehensive options trading platform that combines AI-driven market intelligence with rigorous quantitative analysis and professional-grade backtesting. It implements a sophisticated **"Buy 1.5yr, Sell 1yr"** LEAP options strategy while maintaining strict temporal integrity to prevent look-ahead bias.

### Key Features

- **🤖 AI-Powered Discovery**: Uses Grok to scan news and social media for trending investment themes
- **📊 Automated Options Analysis**: Fetches real-time option chains and Greeks via yfinance
- **🧮 Chain-of-Thought Reasoning**: DeepSeek R1 performs deep quantitative analysis with **real-time streaming output**
- **🛡️ Risk Assessment**: DeepSeek V3 acts as "devil's advocate" to identify red flags
- **🤖 Multi-LLM Consensus**: Sequential analysis with multiple providers (OpenRouter, Gemini) building on each other's insights
- **🧠 Intelligent Model Selection**: User-configurable sentiment models with early stopping based on confidence thresholds
- **🛠️ Development Mode**: Mock responses for testing without burning API credits
- **📈 Professional Backtesting**: Temporal-constrained historical simulation with realistic assumptions
- **💾 Complete Audit Trail**: SQLite database tracks all recommendations, AI reasoning, and trade performance
- **📄 Clear Trade Recommendations**: Generates BUY/SELL recommendations with detailed markdown reports
- **🎯 Batch Analysis**: Process multiple symbols simultaneously with clear recommendation summaries
- **🔄 Hybrid Approach**: Combines quantitative rigor with AI insights, with graceful fallback to pure analysis

## 🏗️ Architecture

The system features **temporal-constrained AI reasoning** and **professional backtesting capabilities**:

```
┌─────────────────────────────────────────────────────────┐
│                 Orchestrator                            │
│          (Workflow & State Management)                  │
└─────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Discovery  │ │   Market    │ │    Quant    │ │    Risk     │
│    Scout    │ │    Data     │ │  Reasoning  │ │   Critic    │
│   (Grok)    │ │  (yfinance) │ │(DeepSeek R1)│ │(DeepSeek V3)│
│             │ │             │ │  Streaming  │ │             │
│ Temporal    │ │ Real-time   │ │ Chain-of-   │ │ Devil's     │
│ Constrained │ │ Options     │ │ Thought     │ │ Advocate    │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
         │              │              │              │
         └──────────────┴──────────────┴──────────────┴──────────────┘
                            │              │
                            ▼              ▼
                    ┌───────────────┐ ┌───────────────┐
                    │   Backtesting │ │   Database    │
                    │   Engine      │ │   (SQLite)    │
                    │               │ │               │
                    │ Temporal      │ │ Audit Trail   │
                    │ Constraints   │ │ & Performance │
                    └───────────────┘ └───────────────┘
                            │              │
                            └──────────────┴──────────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │   Output Engine     │
                            │                     │
                            │ BUY/SELL            │
                            │ Recommendations     │
                            └─────────────────────┘
```

### Core Components

**🤖 AI Agent Layer**
- **Discovery Scout**: Scans for trending assets with temporal awareness
- **Quant Reasoning Engine**: Deep quantitative analysis with scenario modeling
- **Risk Critic**: Comprehensive risk assessment and red flag identification
- **Market Data Client**: Real-time options data and technical analysis
- **LLM-Enhanced Selector**: Optional AI-powered analysis with sentiment and catalyst detection

**📈 Backtesting Layer**
- **Temporal Constraints**: Prevents look-ahead bias in historical simulations
- **Realistic Assumptions**: Transaction costs, slippage, position sizing
- **Performance Analytics**: Sharpe ratio, drawdown, win rates

**💾 Persistence Layer**
- **SQLite Database**: Complete audit trail of recommendations and reasoning
- **Trade Cards**: Detailed markdown reports with full analysis
- **Performance Tracking**: Historical vs. projected return comparisons

**🔧 Analysis Modes**
- **Pure Quantitative**: Rule-based scoring with weighted factors (Delta, DTE, IV, Liquidity)
- **LLM-Enhanced**: Quantitative analysis + AI insights (sentiment, catalysts, confidence)
- **Hybrid Approach**: Best of both worlds with graceful fallback to quantitative analysis

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- OpenRouter API key ([Get one here](https://openrouter.ai))
- Grok API key (optional, for LLM-enhanced analysis - [Get one here](https://console.x.ai/))

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
# Edit .env and add your API keys:
# - OPENROUTER_API_KEY (required for orchestrator.py)
# - GROK_API_KEY (optional, for LLM-enhanced analysis in main.py)
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

### Mode 3: Enhanced Symbol Analysis with Multi-LLM Consensus

Analyze specific symbols with AI-powered multi-LLM consensus:

```bash
# Standard quantitative analysis
python main.py --symbols NVDA,AAPL,MSFT --min-score 0.65 --max-iv-percentile 85

# Multi-LLM consensus analysis (default: OpenRouter + Gemini)
python main.py --symbols NVDA,AAPL,MSFT --llm-enhanced on --min-score 0.65 --max-iv-percentile 85

# Custom providers and sentiment model
python main.py --symbols NVDA --llm-enhanced on --llm-providers openrouter,gemini --llm-sentiment-model deepseek/deepseek-r1

# Development mode (no API calls)
python main.py --symbols NVDA --llm-enhanced on --llm-providers openrouter,gemini --dev-mode
```

**Enhanced Analysis Options:**
- `--symbols "NVDA,AAPL,MSFT"`: Comma-separated list of symbols to analyze
- `--llm-enhanced on/off`: Enable multi-LLM consensus analysis (default: off)
- `--llm-providers "openrouter,gemini"`: Comma-separated LLM providers (default: openrouter,gemini)
- `--llm-sentiment-model "x-ai/grok-beta"`: OpenRouter model for sentiment analysis (default: grok-beta)
- `--dev-mode`: Use mock responses instead of real API calls (for development)
- `--min-score 0.65`: Minimum quantitative score threshold (default: 0.65)
- `--max-iv-percentile 85`: Maximum implied volatility percentile (default: 85)ore 0.65`: Minimum weighted score threshold (default: 0.65)
- `--max-iv-percentile 85`: Maximum implied volatility percentile (default: 70)
- `--target-delta 0.7`: Target delta for LEAP selection (default: 0.7)
- `--regime-filter on/off`: Enable market regime filtering (default: on)
- `--portfolio-size 250000`: Portfolio size for allocation calculations
- `--export`: Export results to markdown file
- `--verbose`: Detailed output with filtering and scoring debug

**Multi-LLM Consensus Features (when `--llm-enhanced on`):**
- **🤖 Sequential Analysis**: Multiple LLMs analyze sequentially, each building on previous insights
- **🧠 Consensus Building**: LLMs can AGREE, DISAGREE, or REFINE previous analysis
- **🎯 Early Stopping**: Stops when confidence threshold reached (default: 0.85) or max iterations (default: 3)
- **📊 Confidence-Weighted Scoring**: Higher confidence LLMs have more influence on final score
- **🔄 Provider Flexibility**: Support for OpenRouter, Gemini, and future providers
- **�️ Development Mode**: Mock responses for testing without API costs
- **📈 Enhanced Scoring**: Combines quantitative analysis with AI sentiment and catalyst detection
- **📊 Dynamic Scoring**: Adjusts quantitative scores based on AI insights
- **🛡️ Risk Factor Analysis**: AI identifies company-specific and market risks
- **💡 Confidence Scoring**: LLM provides confidence levels for recommendations
- **📝 Natural Language Explanations**: AI-generated narratives for investment theses

**Example - Enhanced MAG7 Analysis:**
```bash
python main.py --symbols AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA \
    --llm-enhanced on \
    --min-score 0.65 \
    --max-iv-percentile 85 \
    --target-delta 0.7 \
    --regime-filter on \
    --portfolio-size 250000 \
    --export \
    --verbose
```

### Mode 4: Export Approved Trades

Export all approved recommendations to markdown:

```bash
python main.py --export
```

### Mode 5: Backtesting (Temporal-Constrained Historical Simulation)

Run professional-grade LEAP options backtesting with temporal constraints to prevent look-ahead bias:

```bash
python main.py --backtest
```

**Backtesting Options:**
- `--backtest-symbols "NVDA,AAPL,MSFT,TSLA,GOOGL"`: Universe of symbols (default)
- `--backtest-start "2020-01-01"`: Start date (YYYY-MM-DD, default: 2020-01-01)
- `--backtest-end "2024-01-01"`: End date (YYYY-MM-DD, default: 2024-01-01)
- `--backtest-capital 100000`: Initial capital in USD (default: $100,000)

**Backtesting Features:**
- **Temporal Constraints**: AI models only use information available up to each historical date
- **Realistic Assumptions**: Transaction costs, bid-ask spreads, position sizing
- **Comprehensive Metrics**: Sharpe ratio, max drawdown, win rate, annualized returns
- **Trade Audit Trail**: Complete log of all executed trades with P&L

**Example - Backtest NVDA from 2023:**
```bash
python main.py --backtest --backtest-symbols "NVDA" --backtest-start "2023-01-01" --backtest-end "2023-12-31" --backtest-capital 50000
```

**Default Backtest Parameters:**
- **Symbols**: NVDA, AAPL, MSFT, TSLA, GOOGL
- **Period**: 2020-01-01 to 2024-01-01
- **Capital**: $100,000
- **Position Size**: 20% of capital per trade
- **Max Positions**: 5 concurrent
- **Strategy**: LEAP Call Options (1.5-2 year expirations)

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

## 🤖 LLM-Enhanced Analysis Features

When using `--llm-enhanced on`, the system integrates Grok AI for advanced market intelligence:

### Enhanced Scoring Model

The LLM-enhanced selector combines quantitative analysis with AI insights:

```
Base Quantitative Score: 0.750
├── Sentiment Adjustment: +0.10 (positive market sentiment)
├── Confidence Factor: 1.15 (high confidence in catalysts)
└── Final Enhanced Score: 0.863
```

### AI-Powered Insights

**📊 Sentiment Analysis**
- Real-time market sentiment scoring (0.0-1.0)
- Social media trend analysis
- News sentiment aggregation
- Market mood indicators

**🚀 Catalyst Detection**
- Upcoming earnings calls
- Product launches and events
- Analyst recommendation changes
- Sector rotation patterns

**🛡️ Risk Factor Analysis**
- Company-specific risks
- Market-wide concerns
- Regulatory considerations
- Competitive landscape changes

**💡 Confidence Scoring**
- LLM confidence in analysis (0.0-1.0)
- Data quality assessment
- Market condition certainty
- Recommendation reliability

### Example LLM-Enhanced Output

```bash
NVDA: STRONG LEAP BUY
  Score: 0.863
  Contract: $75.0 Call Jan 2028
  Delta: 0.85 | Risk: LOW
  🤖 Enhanced Score: 0.863 (Original: 0.750)
  🤖 Confidence Factor: 1.15
  📊 Sentiment: 0.82 (Strong positive sentiment from AI infrastructure demand)
  🚀 Catalysts: AI chip demand, data center expansion, new product launches
```

### API Requirements

**For LLM-Enhanced Analysis:**
- **GROK_API_KEY**: Required for sentiment analysis and catalyst detection
- **Fallback**: Gracefully degrades to quantitative analysis if API unavailable

**Configuration:**
```bash
# .env file
GROK_API_KEY=your_grok_api_key_here
```

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
