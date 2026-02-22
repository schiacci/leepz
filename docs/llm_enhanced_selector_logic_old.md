# LLM-Enhanced Option Selector: Technical Architecture & Logic

## Overview

The `LLMEnhancedOptionSelector` is a sophisticated hybrid analysis system that combines quantitative option scoring with AI-powered market intelligence. This document details the complete logic flow from main.py integration through the enhanced analysis pipeline.

## Architecture Diagram

```
main.py Analysis Flow
├── Market Data Fetch (yfinance)
├── Quantitative Scoring (OptionSelector)
└── LLM Enhancement (if --llm-enhanced on)
    ├── Sentiment Analysis (Grok)
    ├── Catalyst Detection
    ├── Risk Factor Analysis
    ├── Confidence Scoring
    └── Dynamic Score Adjustment
```

## Integration Logic (main.py)

### Command Flow
```python
# main.py lines 280-289
option_rec = selector.recommend_optimal_structure(
    ticker=symbol,
    market_data=market_data,  # Pre-fetched market data
    target_delta=args.target_delta,
    min_dte=180,  # 6 months minimum
    max_dte=1095,  # 3 years maximum
    max_iv_percentile=args.max_iv_percentile,
    risk_tolerance=args.risk_tolerance
)
```

### Selector Selection Logic
```python
# main.py lines 271-278
if args.llm_enhanced == "on":
    from llm_enhanced_selector import LLMEnhancedOptionSelector
    selector = LLMEnhancedOptionSelector()
    print(f"  🤖 Using LLM-Enhanced Analysis")
else:
    from option_selector import OptionSelector
    selector = OptionSelector()
    print(f"  📊 Using Quantitative Analysis")
```

## LLM-Enhanced Selector Logic

### 1. Initialization & Setup
```python
class LLMEnhancedOptionSelector:
    def __init__(self):
        self.quant_selector = QuantOptionSelector()  # Baseline quantitative engine
        self.grok_api_key = os.getenv('GROK_API_KEY')  # Optional API key
        self.base_url = "https://api.x.ai/v1/chat/completions"
```

**Key Design Principles:**
- **Graceful Degradation**: Works without API key, falls back to pure quantitative
- **Hybrid Approach**: Combines quantitative rigor with AI insights
- **Modular Design**: Can be easily extended with additional LLM providers

### 2. Main Recommendation Flow

#### Step 1: Quantitative Baseline
```python
def recommend_optimal_structure(self, ticker, market_data, ...):
    # Get quantitative baseline first
    quant_result = self.quant_selector.recommend_optimal_structure(
        ticker, market_data, target_delta, min_dte, max_dte, max_iv_percentile, risk_tolerance
    )
    
    if not quant_result.get('success'):
        return quant_result  # Early return if quantitative fails
```

**Logic Rationale:**
- Quantitative analysis provides the foundation
- Ensures system works even without LLM availability
- Maintains mathematical rigor and reproducibility

#### Step 2: LLM Insights Gathering
```python
def _get_llm_insights(self, ticker: str, market_data: MarketData):
    # API key check with graceful fallback
    if not self.grok_api_key:
        print(f"  ⚠️ No GROK_API_KEY found, using baseline analysis")
        return {}
    
    # Market context collection
    regime = get_market_regime()
    stock = yf.Ticker(ticker)
    hist = stock.history(period="30d")
    current_price = hist['Close'].iloc[-1]
    price_change_7d = ((current_price - hist['Close'].iloc[-8]) / hist['Close'].iloc[-8]) * 100
    price_change_30d = ((current_price - hist['Close'].iloc[-30]) / hist['Close'].iloc[-30]) * 100
```

**Context Building Strategy:**
- **Market Regime**: RISK_ON/TRANSITION/RISK_OFF affects sentiment interpretation
- **Price Momentum**: 7-day and 30-day changes provide trend context
- **Volatility Context**: IV percentile helps assess option pricing environment
- **Contract Availability**: Number of available contracts indicates liquidity

#### Step 3: LLM Prompt Engineering
```python
prompt = f"""
Analyze {ticker} for LEAP options investment and provide insights:

Current Market Context:
- Stock Price: ${current_price:.2f} (7d: {price_change_7d:+.1f}%, 30d: {price_change_30d:+.1f}%)
- Market Regime: {regime}
- IV Percentile: {market_data.implied_volatility_rank:.1f}%
- Available Contracts: {len(market_data.option_contracts)}

Provide analysis in JSON format:
{{
    "sentiment_score": 0.0-1.0,
    "sentiment_reasoning": "brief explanation",
    "volatility_outlook": "LOW/MEDIUM/HIGH",
    "optimal_delta_adjustment": -0.1 to 0.1,
    "optimal_dte_adjustment": -180 to 180,
    "confidence_level": 0.0-1.0,
    "key_catalysts": ["catalyst1", "catalyst2"],
    "risk_factors": ["risk1", "risk2"]
}}

Focus on:
- Recent news sentiment and social media trends
- Technical analysis of price action
- Volatility expectations
- Upcoming catalysts (earnings, product launches, etc.)
- Risk factors specific to this company
"""
```

**Prompt Design Principles:**
- **Structured Output**: JSON format ensures consistent parsing
- **Quantified Insights**: All outputs are numeric or structured lists
- **Context-Aware**: Includes current market conditions
- **Actionable Focus**: Provides specific adjustments for option selection

#### Step 4: API Integration & Error Handling
```python
# Grok API call with robust error handling
headers = {
    "Authorization": f"Bearer {self.grok_api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "grok-beta",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 500,
    "temperature": 0.3  # Low temperature for consistent outputs
}

response = requests.post(self.base_url, headers=headers, json=data, timeout=10)

if response.status_code == 200:
    result = response.json()
    llm_output = result['choices'][0]['message']['content']
    
    # Robust JSON parsing with fallback
    try:
        insights = json.loads(llm_output)
        print(f"  🤖 LLM Insights: Sentiment {insights.get('sentiment_score', 0.5):.2f}")
        return insights
    except json.JSONDecodeError:
        print(f"  ⚠️ LLM response parsing failed")
        return {}
else:
    print(f"  ⚠️ LLM API call failed: {response.status_code}")
    return {}
```

**Error Handling Strategy:**
- **Network Resilience**: 10-second timeout prevents hanging
- **Parsing Robustness**: JSON errors don't crash the system
- **Graceful Degradation**: Always returns to quantitative baseline
- **User Feedback**: Clear error messages for debugging

### 3. Score Enhancement Logic

#### Step 1: Base Score Extraction
```python
def _apply_llm_enhancements(self, quant_result, llm_insights, market_data, ticker):
    enhanced_result = quant_result.copy()
    
    # Extract original components
    orig_rec = quant_result['recommendation']
    orig_contract = orig_rec['contract']
    orig_score = orig_rec['score']
```

#### Step 2: Sentiment-Based Adjustments
```python
# Apply LLM adjustments
sentiment_score = llm_insights.get('sentiment_score', 0.5)
delta_adjustment = llm_insights.get('optimal_delta_adjustment', 0.0)
dte_adjustment = llm_insights.get('optimal_dte_adjustment', 0.0)
confidence = llm_insights.get('confidence_level', 0.5)

# Enhanced scoring with LLM insights
enhanced_score = orig_score

# Sentiment boost/penalty
if sentiment_score > 0.7:
    enhanced_score += 0.1  # Boost for positive sentiment
    print(f"  📈 Sentiment boost: +0.10 (positive sentiment)")
elif sentiment_score < 0.3:
    enhanced_score -= 0.1  # Penalty for negative sentiment
    print(f"  📉 Sentiment penalty: -0.10 (negative sentiment)")
```

**Sentiment Logic:**
- **> 0.7**: Strong positive sentiment → +0.10 score boost
- **< 0.3**: Strong negative sentiment → -0.10 score penalty
- **0.3-0.7**: Neutral sentiment → no adjustment

#### Step 3: Confidence-Based Scaling
```python
# Confidence adjustment
confidence_factor = 0.8 + (confidence * 0.4)  # 0.8 to 1.2 range
enhanced_score *= confidence_factor

print(f"  🤖 Enhanced Score: {enhanced_score:.3f} (confidence: {confidence:.2f})")
```

**Confidence Scaling Formula:**
```
confidence_factor = 0.8 + (confidence * 0.4)
# Range: 0.8 (low confidence) to 1.2 (high confidence)
# Neutral confidence (0.5) = 1.0 (no scaling)
```

#### Step 4: Result Integration
```python
# Update recommendation with enhanced data
enhanced_result['recommendation'].update({
    'enhanced_score': enhanced_score,
    'llm_insights': llm_insights,
    'sentiment_adjustment': sentiment_score - 0.5,  # Deviation from neutral
    'confidence_factor': confidence_factor,
    'original_score': orig_score
})

# Add LLM narrative
narrative = self._generate_llm_narrative(ticker, llm_insights, orig_contract)
enhanced_result['llm_narrative'] = narrative
```

### 4. Narrative Generation

#### Natural Language Explanation
```python
def _generate_llm_narrative(self, ticker, llm_insights, contract):
    sentiment = llm_insights.get('sentiment_score', 0.5)
    catalysts = llm_insights.get('key_catalysts', [])
    risks = llm_insights.get('risk_factors', [])
    
    narrative = f"LLM Analysis for {ticker}:\n"
    
    # Sentiment analysis
    if sentiment > 0.7:
        narrative += f"• Strong positive sentiment ({sentiment:.1f}) suggests favorable market conditions\n"
    elif sentiment < 0.3:
        narrative += f"• Negative sentiment ({sentiment:.1f}) indicates caution advised\n"
    else:
        narrative += f"• Neutral sentiment ({sentiment:.1f}) with balanced risk/reward\n"
    
    # Catalysts and risks
    if catalysts:
        narrative += f"• Key catalysts: {', '.join(catalysts[:2])}\n"
    if risks:
        narrative += f"• Risk factors: {', '.join(risks[:2])}\n"
    
    # Contract specifics
    narrative += f"• Selected contract: ${contract.strike} Call expiring {contract.expiration.strftime('%b %Y')}\n"
    narrative += f"• Delta: {contract.delta:.2f}, IV: {contract.implied_volatility*100:.1f}%\n"
    
    return narrative
```

## Complete Analysis Flow

### Input Parameters
```python
# From main.py
ticker: str                    # Stock symbol
market_data: MarketData        # Pre-fetched option chain
target_delta: float = 0.7      # Desired option delta
min_dte: int = 180            # Minimum days to expiration
max_dte: int = 1095           # Maximum days to expiration
max_iv_percentile: float = 70.0  # IV filtering threshold
risk_tolerance: str = "MODERATE"  # Risk preference
```

### Processing Pipeline
```
1. Quantitative Analysis (OptionSelector)
   ├── Contract filtering (DTE, delta, IV, liquidity)
   ├── Weighted scoring (6-factor model)
   └── Best contract selection

2. LLM Enhancement (if API key available)
   ├── Market context collection
   ├── Grok API call with structured prompt
   ├── JSON response parsing
   └── Insight extraction

3. Score Enhancement
   ├── Sentiment-based adjustments (+/- 0.10)
   ├── Confidence scaling (0.8-1.2x multiplier)
   └── Final enhanced score calculation

4. Output Generation
   ├── Enhanced recommendation object
   ├── LLM insights integration
   ├── Natural language narrative
   └── Fallback to quantitative if LLM fails
```

### Output Structure
```python
{
    'success': True,
    'recommendation': {
        'contract': OptionContract,
        'score': 0.750,                    # Original quantitative score
        'enhanced_score': 0.863,          # LLM-enhanced score
        'llm_insights': {
            'sentiment_score': 0.82,
            'sentiment_reasoning': "Strong AI infrastructure demand",
            'key_catalysts': ["AI chip demand", "data center expansion"],
            'risk_factors': ["Regulatory concerns", "Competition"],
            'confidence_level': 0.85
        },
        'sentiment_adjustment': 0.32,      # Sentiment - 0.5 (neutral)
        'confidence_factor': 1.14,         # Applied multiplier
        'original_score': 0.750
    },
    'llm_narrative': "Natural language explanation..."
}
```

## Design Benefits

### 1. Hybrid Intelligence
- **Quantitative Foundation**: Mathematical rigor and reproducibility
- **AI Enhancement**: Market sentiment and catalyst detection
- **Graceful Degradation**: Works without API keys

### 2. Robust Error Handling
- **Network Resilience**: Timeouts and connection failures
- **Parsing Safety**: JSON errors don't crash system
- **User Feedback**: Clear debugging information

### 3. Modular Architecture
- **Pluggable LLMs**: Easy to add other AI providers
- **Separable Components**: Can use quantitative only
- **Extensible**: New insight types easily added

### 4. Performance Optimization
- **Single API Call**: Efficient LLM usage
- **Caching Ready**: Market data reused across symbols
- **Batch Processing**: Handles multiple symbols efficiently

## Usage Examples

### Basic Quantitative Analysis
```bash
python main.py --symbols NVDA --min-score 0.65 --max-iv-percentile 85
# Output: Pure quantitative scoring
```

### LLM-Enhanced Analysis
```bash
python main.py --symbols NVDA --llm-enhanced on --min-score 0.65 --max-iv-percentile 85
# Output: Quantitative + AI insights + enhanced scoring
```

### Enhanced Output Example
```
NVDA: STRONG LEAP BUY
  Score: 0.863
  Contract: $75.0 Call Jan 2028
  Delta: 0.85 | Risk: LOW
  🤖 Enhanced Score: 0.863 (Original: 0.750)
  🤖 Confidence Factor: 1.14
  📊 Sentiment: 0.82 (Strong positive sentiment from AI infrastructure demand)
  🚀 Catalysts: AI chip demand, data center expansion, new product launches
```

This architecture provides a robust, extensible foundation for AI-enhanced options analysis while maintaining the mathematical rigor required for financial decision-making.
