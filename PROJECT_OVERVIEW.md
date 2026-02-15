# LEAP Strategic Asset Engine - Project Overview

## 📋 What You Received

A complete, production-ready Python application for discovering and analyzing LEAP options opportunities using AI.

## 🗂️ File Structure

```
leap_engine/
├── 📄 README.md              # Comprehensive documentation
├── 📄 SETUP.md               # Quick setup guide
├── 📄 requirements.txt       # Python dependencies
├── 📄 .env.example          # Environment configuration template
├── 📄 .gitignore            # Git ignore rules
│
├── 🐍 main.py               # Main entry point (CLI)
├── 🐍 orchestrator.py       # Workflow controller
├── 🐍 config.py             # Configuration management
├── 🐍 examples.py           # Usage examples
│
├── agents/
│   ├── __init__.py
│   └── 🤖 ai_agents.py      # Discovery, Reasoning, Critique
│
├── data/
│   ├── __init__.py
│   └── 📊 market_data_client.py  # yfinance integration
│
├── database/
│   ├── __init__.py
│   └── 💾 db.py             # SQLite persistence
│
└── models/
    └── 📦 __init__.py       # Pydantic data models
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup environment:**
   ```bash
   cp .env.example .env
   # Add your OPENROUTER_API_KEY
   ```

3. **Run a test:**
   ```bash
   python main.py --ticker NVDA --narrative "AI chip leader"
   ```

## 🎯 Core Components

### 1. DiscoveryScout (agents/ai_agents.py)
- Uses Grok via OpenRouter to scan for trending assets
- Identifies sectors and high-conviction tickers
- Returns structured TrendingAsset objects

### 2. MarketDataClient (data/market_data_client.py)
- Fetches real-time market data via yfinance
- Downloads option chains with Greeks
- Filters for LEAP criteria (540+ days, 0.80 delta)

### 3. QuantReasoningEngine (agents/ai_agents.py)
- Uses DeepSeek R1 for Chain-of-Thought analysis
- Performs 3-point scenario analysis
- Calculates expected values and projections

### 4. RiskCritic (agents/ai_agents.py)
- Uses DeepSeek V3 as devil's advocate
- Checks IV levels, liquidity, earnings
- Makes final approve/reject decision

### 5. Orchestrator (orchestrator.py)
- Manages the complete workflow
- Coordinates between all components
- Generates Trade Cards and saves to database

### 6. Database (database/db.py)
- SQLite for persistence
- Stores recommendations and audit trail
- Tracks AI reasoning for each decision

## 📊 LEAP Strategy Logic

The system implements a **"Buy 1.5yr, Sell at 1yr"** strategy:

1. **Entry Criteria:**
   - Time to expiry: 540+ days (1.5 years minimum)
   - Delta: 0.80 (deep in-the-money for stock-like exposure)
   - IV Percentile: < 40 (avoid elevated volatility)
   - Bid/Ask Spread: < 5% (ensure liquidity)

2. **Exit Strategy:**
   - Hold until Day 365 (1 year mark)
   - Leaves 6 months of extrinsic value remaining
   - Avoids the "Theta Cliff" near expiration

3. **Risk Management:**
   - Max loss = premium paid
   - Position sized based on conviction
   - Critical issues = automatic rejection

## 🔑 Key Features

### Modular Design
- Each component is independent and testable
- Easy to swap AI models or data sources
- Configuration-driven behavior

### Type Safety
- Pydantic models for all data structures
- Runtime validation of inputs
- Clear error messages

### Audit Trail
- Every AI decision is logged
- Complete reasoning stored in database
- Reproducible analysis

### Extensible
- Add custom risk checks easily
- Plug in alternative data sources
- Customize output formats

## 📈 Usage Modes

### Mode 1: Full Discovery Pipeline
```bash
python main.py --sectors 5 --tickers-per-sector 2
```
Discovers trending assets and analyzes them automatically.

### Mode 2: Manual Ticker Analysis
```bash
python main.py --ticker AAPL --narrative "Your thesis here"
```
Analyze a specific ticker with custom narrative.

### Mode 3: Export Results
```bash
python main.py --export
```
Export all approved trades to markdown files.

### Mode 4: Programmatic (examples.py)
```python
from orchestrator import Orchestrator

orchestrator = Orchestrator()
trade_card = orchestrator.analyze_single_ticker("MSFT", "Cloud growth thesis")
```

## 🛠️ Customization Points

### 1. LEAP Parameters (.env)
```bash
MIN_DAYS_TO_EXPIRY=540
TARGET_DELTA=0.80
MAX_IV_PERCENTILE=40
```

### 2. AI Models (.env)
```bash
GROK_MODEL=x-ai/grok-beta
DEEPSEEK_R1_MODEL=deepseek/deepseek-r1
DEEPSEEK_V3_MODEL=deepseek/deepseek-chat
```

### 3. Risk Checks (agents/ai_agents.py)
Modify `RiskCritic._build_critique_prompt()` to add custom checks.

### 4. Data Sources (data/market_data_client.py)
Extend `MarketDataClient` to add alternative APIs.

## 📚 Files to Read First

1. **SETUP.md** - Get up and running quickly
2. **README.md** - Full documentation
3. **main.py** - See the CLI interface
4. **orchestrator.py** - Understand the workflow
5. **models/__init__.py** - See the data structures

## ⚠️ Important Notes

- **Not Financial Advice**: This is a research tool
- **Verify Data**: Always check quotes in your broker
- **Test First**: Paper trade before using real capital
- **API Costs**: OpenRouter charges per token (DeepSeek is very cheap)
- **Rate Limits**: yfinance may rate limit on heavy usage

## 🎓 Learning Path

1. Read SETUP.md and configure the project
2. Run a single ticker analysis to see the output
3. Review the generated Trade Card markdown
4. Check the database to see stored data
5. Modify LEAP parameters and compare results
6. Extend with your own custom checks

## 🔗 Dependencies

- **yfinance**: Market data and option chains
- **OpenRouter**: Unified LLM API (Grok, DeepSeek)
- **Pydantic**: Data validation and type safety
- **SQLAlchemy**: Database ORM
- **pandas/numpy**: Data manipulation

## 🚀 Next Steps

1. **Setup**: Follow SETUP.md to configure
2. **Test**: Run with a known ticker (AAPL, MSFT, etc.)
3. **Explore**: Check the database and logs
4. **Customize**: Adjust parameters in .env
5. **Extend**: Add your own risk checks or data sources

## 💡 Tips

- Start with liquid, well-known tickers
- Review the AI's reasoning in the database
- Don't blindly trust approvals - read the critique
- Track actual performance vs. projections
- Adjust conviction thresholds based on results

---

**Built by Claude** based on your system design specifications.

This is a fully functional, production-ready codebase ready for further development and backtesting.
