"""
Data models for LEAP Strategic Asset Engine
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class AssetType(str, Enum):
    """Type of asset"""
    STOCK = "stock"
    CRYPTO = "crypto"


class TrendingAsset(BaseModel):
    """Asset identified by discovery scout"""
    ticker: str
    sector: str
    narrative: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    source: str  # X, news, etc.
    discovered_at: datetime = Field(default_factory=datetime.now)


class OptionContract(BaseModel):
    """LEAP option contract details"""
    ticker: str
    strike: float
    expiration: datetime
    days_to_expiry: int
    option_type: str = "call"  # We focus on calls for LEAP strategy
    
    # Pricing
    last_price: float
    bid: float
    ask: float
    bid_ask_spread_pct: float
    
    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_volatility: Optional[float] = None
    
    # Metadata
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    in_the_money: bool = False


class MarketData(BaseModel):
    """Market data for an asset"""
    ticker: str
    spot_price: float
    
    # Volatility metrics
    implied_volatility_rank: Optional[float] = None  # IV percentile
    historical_volatility: Optional[float] = None
    
    # Option chain
    option_contracts: List[OptionContract] = Field(default_factory=list)
    
    # Price history for analysis
    price_52w_high: Optional[float] = None
    price_52w_low: Optional[float] = None
    
    fetched_at: datetime = Field(default_factory=datetime.now)


class ScenarioAnalysis(BaseModel):
    """3-point scenario analysis for LEAP valuation"""
    bull_case: Dict[str, float]  # +20% move
    base_case: Dict[str, float]  # +5% move
    bear_case: Dict[str, float]  # -15% move
    expected_value: float
    expected_return_pct: float
    risk_reward_ratio: float


class ReasoningOutput(BaseModel):
    """Output from the Quant Reasoning Engine"""
    ticker: str
    recommended_contract: Optional[OptionContract] = None
    scenario_analysis: Optional[ScenarioAnalysis] = None
    
    # AI reasoning
    chain_of_thought: str
    conviction_score: float = Field(ge=0.0, le=1.0)
    
    # Key metrics
    break_even_price: Optional[float] = None
    max_loss: Optional[float] = None
    projected_1yr_value: Optional[float] = None
    
    reasoning_timestamp: datetime = Field(default_factory=datetime.now)


class RiskFlags(BaseModel):
    """Risk assessment flags"""
    high_iv: bool = False
    low_liquidity: bool = False
    wide_spread: bool = False
    upcoming_earnings: bool = False
    insufficient_time_value: bool = False
    
    critical_issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class CritiqueOutput(BaseModel):
    """Output from the Risk Critic"""
    ticker: str
    approved: bool
    risk_flags: RiskFlags
    
    # Devil's advocate reasoning
    critique_reasoning: str
    risk_score: float = Field(ge=0.0, le=1.0)
    
    critique_timestamp: datetime = Field(default_factory=datetime.now)


class TradeCard(BaseModel):
    """Final trade recommendation card"""
    # Asset info
    ticker: str
    sector: str
    discovery_narrative: str
    
    # Option details
    contract: OptionContract
    
    # Entry & Exit
    entry_price: float
    projected_exit_price_1yr: float
    projected_return_pct: float
    
    # Risk metrics
    max_loss: float
    break_even_price: float
    risk_score: float
    
    # Scenarios
    scenario_analysis: ScenarioAnalysis
    
    # Flags
    risk_flags: RiskFlags
    approved: bool
    
    # AI reasoning (for audit trail)
    quant_reasoning: str
    risk_critique: str
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    
    def to_markdown(self) -> str:
        """Convert trade card to readable markdown"""
        md = f"""# LEAP Trade Card: {self.ticker}

## 📊 Overview
- **Sector**: {self.sector}
- **Narrative**: {self.discovery_narrative}
- **Approved**: {'✅ YES' if self.approved else '❌ NO'}

## 🎯 Option Contract
- **Strike**: ${self.contract.strike:.2f}
- **Expiration**: {self.contract.expiration.strftime('%Y-%m-%d')} ({self.contract.days_to_expiry} days)
- **Delta**: {self.contract.delta:.3f}
- **Entry Price**: ${self.entry_price:.2f}
- **Bid/Ask Spread**: {self.contract.bid_ask_spread_pct:.2f}%

## 💰 Projected Returns (1-Year Exit)
- **Exit Price**: ${self.projected_exit_price_1yr:.2f}
- **Return**: {self.projected_return_pct:.1f}%
- **Break-Even**: ${self.break_even_price:.2f}
- **Max Loss**: ${self.max_loss:.2f}

## 📈 Scenario Analysis
- **Bull (+20%)**: ${self.scenario_analysis.bull_case.get('option_value', 0):.2f} ({self.scenario_analysis.bull_case.get('return_pct', 0):.1f}%)
- **Base (+5%)**: ${self.scenario_analysis.base_case.get('option_value', 0):.2f} ({self.scenario_analysis.base_case.get('return_pct', 0):.1f}%)
- **Bear (-15%)**: ${self.scenario_analysis.bear_case.get('option_value', 0):.2f} ({self.scenario_analysis.bear_case.get('return_pct', 0):.1f}%)
- **Expected Value**: ${self.scenario_analysis.expected_value:.2f}

## ⚠️ Risk Assessment
**Risk Score**: {self.risk_score:.2f}/1.0

**Flags**:
{self._format_risk_flags()}

## 🤖 AI Analysis
### Quant Reasoning
{self.quant_reasoning}

### Risk Critique
{self.risk_critique}

---
*Generated: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return md
    
    def _format_risk_flags(self) -> str:
        """Format risk flags for display"""
        flags = []
        if self.risk_flags.high_iv:
            flags.append("- ⚠️ High Implied Volatility")
        if self.risk_flags.low_liquidity:
            flags.append("- ⚠️ Low Liquidity")
        if self.risk_flags.wide_spread:
            flags.append("- ⚠️ Wide Bid/Ask Spread")
        if self.risk_flags.upcoming_earnings:
            flags.append("- ⚠️ Upcoming Earnings")
        if self.risk_flags.insufficient_time_value:
            flags.append("- ⚠️ Insufficient Time Value")
        
        return "\n".join(flags) if flags else "- ✅ No major flags"
