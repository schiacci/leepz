"""
AI Agents for LEAP Strategic Asset Engine
Uses OpenRouter API to access Grok and DeepSeek models
"""
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import requests

from models import (
    TrendingAsset, MarketData, OptionContract, 
    ReasoningOutput, ScenarioAnalysis, CritiqueOutput, RiskFlags
)
from config import config


class OpenRouterClient:
    """Base client for OpenRouter API calls"""
    
    def __init__(self):
        self.api_key = config.openrouter.api_key
        self.base_url = config.openrouter.base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/leap-engine",  # Optional
            "X-Title": "LEAP Strategic Asset Engine"  # Optional
        }
    
    def call_llm(self, model: str, messages: List[Dict[str, str]], 
                 temperature: float = 0.7, max_tokens: int = 4000,
                 stream: bool = False, stream_callback: Optional[callable] = None) -> str:
        """
        Make an API call to OpenRouter
        
        Args:
            model: Model identifier (e.g., "x-ai/grok-beta")
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            stream: Whether to stream the response
            stream_callback: Callback function for streaming chunks
            
        Returns:
            Model's response text
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60,
                stream=stream
            )
            response.raise_for_status()
            
            if stream and stream_callback:
                # Handle streaming response
                print("  🔄 Starting streaming response processing...")
                full_response = ""
                chunk_count = 0
                thought_buffer = ""  # Buffer for complete thoughts
                buffer_chunks = 0  # Count chunks in buffer
                
                for line in response.iter_lines():
                    if line:
                        chunk_count += 1
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            if line.strip() == 'data: [DONE]':
                                # Send any remaining buffered thought
                                if thought_buffer.strip():
                                    stream_callback(thought_buffer)
                                    full_response += thought_buffer
                                print("  🏁 Streaming complete")
                                break
                            try:
                                chunk = json.loads(line[6:])  # Remove 'data: ' prefix
                                if 'choices' in chunk and chunk['choices']:
                                    choice = chunk['choices'][0]
                                    if 'delta' in choice:
                                        delta = choice['delta']
                                        reasoning = delta.get('reasoning', '')
                                        content = delta.get('content', '')
                                        
                                        # DeepSeek R1 streams reasoning in 'reasoning' field, not 'content'
                                        if reasoning:
                                            thought_buffer += reasoning
                                            full_response += reasoning
                                            buffer_chunks += 1
                                            
                                            # Check for sentence endings, section breaks, or reasonable chunk limit
                                            if (any(end in reasoning for end in ['.', '!', '?', '\n\n', '**', '$$']) or 
                                                any(phrase in reasoning for phrase in ['Step ', 'Now ', "Let's ", 'First,', 'Next,', 'Finally,']) or
                                                buffer_chunks >= 200):
                                                stream_callback(thought_buffer)
                                                thought_buffer = ""
                                                buffer_chunks = 0
                                                
                                        elif content:
                                            thought_buffer += content
                                            full_response += content
                                            buffer_chunks += 1
                                            
                                            # Check for sentence endings, line breaks, markdown, or reasonable chunk limit for content
                                            if (any(end in content for end in ['.', '!', '?', '\n', '**', '$$', '\n\n']) or 
                                                buffer_chunks >= 100):
                                                stream_callback(thought_buffer)
                                                thought_buffer = ""
                                                buffer_chunks = 0
                            except json.JSONDecodeError:
                                continue
                
                print(f"  📊 Total chunks processed: {chunk_count}")
                return full_response
            else:
                # Handle non-streaming response
                result = response.json()
                return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            print(f"❌ OpenRouter API error: {e}")
            raise


class DiscoveryScout(OpenRouterClient):
    """
    The "Scout" - discovers trending assets using Grok
    Scans X (Twitter) and news for high-conviction narratives
    """
    
    def __init__(self):
        super().__init__()
        self.model = config.openrouter.grok_model
    
    def discover_trending_assets(self, num_sectors: int = 5, 
                                 tickers_per_sector: int = 2) -> List[TrendingAsset]:
        """
        Discover trending assets across multiple sectors
        
        Args:
            num_sectors: Number of trending sectors to identify
            tickers_per_sector: Number of top tickers per sector
            
        Returns:
            List of TrendingAsset objects
        """
        prompt = f"""You are a market research analyst scanning for high-conviction investment opportunities.

Task: Identify {num_sectors} trending sectors and {tickers_per_sector} top stock tickers per sector that are getting significant attention in financial news and social media (X/Twitter) recently.

For each ticker, provide:
1. The ticker symbol
2. The sector (e.g., "AI Infrastructure", "Semiconductors", "Clean Energy")
3. A concise narrative explaining why it's trending (macro tailwind, catalyst, sentiment)
4. A confidence score (0.0 to 1.0) based on the strength of the narrative

Return your response in JSON format:
```json
[
  {{
    "ticker": "NVDA",
    "sector": "AI Infrastructure",
    "narrative": "Dominant AI chip maker benefiting from enterprise AI adoption and data center buildout",
    "confidence_score": 0.85,
    "source": "X and financial news"
  }},
  ...
]
```

Focus on:
- Stocks with clear macro tailwinds (not just hype)
- Companies with institutional backing
- Assets suitable for LEAP options (liquid, established companies)
- Avoid penny stocks and illiquid names

Current date: {datetime.now().strftime('%Y-%m-%d')}
"""
        
        messages = [
            {"role": "system", "content": "You are an expert market analyst specializing in identifying high-conviction investment themes."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.call_llm(self.model, messages, temperature=0.8)
            
            # Extract JSON from response
            json_str = self._extract_json(response)
            assets_data = json.loads(json_str)
            
            # Convert to TrendingAsset objects
            trending_assets = []
            for asset in assets_data:
                trending_assets.append(TrendingAsset(
                    ticker=asset['ticker'],
                    sector=asset['sector'],
                    narrative=asset['narrative'],
                    confidence_score=asset['confidence_score'],
                    source=asset.get('source', 'AI Discovery')
                ))
            
            print(f"✅ Discovered {len(trending_assets)} trending assets")
            return trending_assets
            
        except Exception as e:
            print(f"❌ Discovery error: {e}")
            return []
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from markdown code blocks or raw text"""
        # Try to find JSON in code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        else:
            # Assume the whole response is JSON
            return text.strip()


class QuantReasoningEngine(OpenRouterClient):
    """
    The "Quant" - performs Chain-of-Thought analysis using DeepSeek R1
    Evaluates LEAP options and projects 1-year exit scenarios
    """
    
    def __init__(self):
        super().__init__()
        self.model = config.openrouter.deepseek_r1_model
        self.leap_heuristics = config.leap_heuristics
    
    def analyze_leap_opportunity(self, ticker: str, market_data: MarketData, 
                                 narrative: str) -> Optional[ReasoningOutput]:
        """
        Perform deep quantitative analysis on a LEAP candidate
        
        Args:
            ticker: Stock symbol
            market_data: Market data with option chain
            narrative: Discovery narrative explaining the thesis
            
        Returns:
            ReasoningOutput with recommendation and scenario analysis
        """
        if not market_data.option_contracts:
            print(f"⚠️ No suitable LEAP contracts found for {ticker}")
            return None
        
        # Select best contract (closest to 0.80 delta with longest expiry)
        best_contract = market_data.option_contracts[0]
        print(f"  📊 Analyzing {ticker}: ${best_contract.strike} Call {best_contract.expiration.strftime('%Y-%m-%d')}")
        print(f"     Delta: {best_contract.delta:.3f} | Mid: ${((best_contract.bid + best_contract.ask) / 2):.2f}")
        print(f"     IV: {best_contract.implied_volatility:.2%} | Days: {best_contract.days_to_expiry}")
        
        print(f"  🧠 Building analysis prompt for {ticker}...")
        prompt = self._build_reasoning_prompt(ticker, market_data, best_contract, narrative)
        print(f"  🤖 Calling DeepSeek R1 for quantitative reasoning...")
        
        messages = [
            {"role": "system", "content": "You are a quantitative analyst specializing in LEAP options strategies. Use rigorous Chain-of-Thought reasoning."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            print(f"  ⏳ Processing {ticker} analysis (this may take 30-60 seconds)...")
            
            # Define streaming callback to show chain of thought in real-time
            def stream_callback(chunk: str):
                # Clean up and format the chunk
                chunk = chunk.strip()
                if not chunk:
                    return
                
                # Detect reasoning vs content based on chunk characteristics
                if any(chunk.endswith(end) for end in ['.', '!', '?']) or '**' in chunk or '$$' in chunk:
                    # Complete thoughts or structured content - show with newlines
                    print(f"🤔 {chunk}")
                    print()  # Add blank line for readability
                elif any(phrase in chunk for phrase in ['Step ', 'Now ', "Let's ", 'First,', 'Next,', 'Finally,']):
                    # Section headers - highlight them
                    print(f"🤔 {chunk}")
                    print()
                else:
                    # Mathematical content or calculations - format inline
                    # Replace some LaTeX-like symbols for better readability
                    formatted = chunk.replace('$$', '').replace('\\', '').replace('frac', '/').replace('left', '').replace('right', '')
                    print(f"� {formatted}", end='', flush=True)
            
            print("  🔄 Attempting streaming analysis...")
            response = self.call_llm(self.model, messages, temperature=0.3, max_tokens=6000, 
                                   stream=True, stream_callback=stream_callback)
            print(f"\n  ✅ Received analysis response for {ticker}")
            
            # Parse the structured response
            print(f"  📋 Parsing quantitative results for {ticker}...")
            reasoning_output = self._parse_reasoning_response(response, ticker, best_contract)
            
            print(f"✅ Completed quant analysis for {ticker}")
            return reasoning_output
            
        except Exception as e:
            print(f"❌ Reasoning error for {ticker}: {e}")
            return None
    
    def _build_reasoning_prompt(self, ticker: str, market_data: MarketData,
                                contract: OptionContract, narrative: str) -> str:
        """Build the detailed reasoning prompt"""
        
        prompt = f"""# LEAP Options Analysis: {ticker}

## Investment Thesis
{narrative}

## Current Market Data
- **Spot Price**: ${market_data.spot_price:.2f}
- **52-Week Range**: ${market_data.price_52w_low:.2f} - ${market_data.price_52w_high:.2f}
- **IV Rank**: {market_data.implied_volatility_rank or 'N/A'}

## Proposed LEAP Contract
- **Strike**: ${contract.strike:.2f}
- **Expiration**: {contract.expiration.strftime('%Y-%m-%d')} ({contract.days_to_expiry} days)
- **Delta**: {contract.delta:.3f}
- **Mid Price**: ${((contract.bid + contract.ask) / 2):.2f}
- **Bid/Ask Spread**: {contract.bid_ask_spread_pct:.2f}%
- **Implied Volatility**: {contract.implied_volatility:.2%}
- **Theta (daily decay)**: {contract.theta}

## Your Task
Perform a rigorous Chain-of-Thought analysis to determine if this is a sound LEAP trade using the "Buy 1.5yr, Sell 1yr" strategy.

### Required Analysis:
1. **Break-Even Calculation**: What stock price is needed at Day 365 to break even?

2. **Scenario Analysis**: Calculate the option's value at the 1-year mark (Day 365) under three scenarios:
   - **Bull Case** (+20% stock move): Stock → ${market_data.spot_price * 1.20:.2f}
   - **Base Case** (+5% stock move): Stock → ${market_data.spot_price * 1.05:.2f}
   - **Bear Case** (-15% stock move): Stock → ${market_data.spot_price * 0.85:.2f}
   
   For each scenario, estimate:
   - Option intrinsic value (Stock Price - Strike)
   - Remaining extrinsic value (with 6 months left)
   - Total option value
   - Return %

3. **Expected Value**: Calculate probability-weighted expected return
   - Assume: Bull 30%, Base 50%, Bear 20%

4. **Risk Assessment**:
   - Maximum loss (premium paid)
   - Risk/Reward ratio
   - Time decay impact

5. **Conviction Score**: Rate 0.0 to 1.0 based on:
   - Thesis strength
   - Options pricing efficiency
   - Risk/reward profile

## Output Format
Provide your analysis in JSON:
```json
{{
  "chain_of_thought": "Your detailed step-by-step reasoning here...",
  "break_even_price": 150.00,
  "scenario_analysis": {{
    "bull_case": {{"stock_price": 180.00, "option_value": 35.00, "return_pct": 40.0}},
    "base_case": {{"stock_price": 157.50, "option_value": 28.00, "return_pct": 12.0}},
    "bear_case": {{"stock_price": 127.50, "option_value": 15.00, "return_pct": -40.0}}
  }},
  "expected_value": 26.50,
  "expected_return_pct": 6.0,
  "max_loss": 25.00,
  "conviction_score": 0.75
}}
```

Be thorough and show your work.
"""
        return prompt
    
    def _parse_reasoning_response(self, response: str, ticker: str, 
                                  contract: OptionContract) -> ReasoningOutput:
        """Parse the JSON response from the reasoning engine"""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            
            # Build scenario analysis
            scenario_analysis = ScenarioAnalysis(
                bull_case=data['scenario_analysis']['bull_case'],
                base_case=data['scenario_analysis']['base_case'],
                bear_case=data['scenario_analysis']['bear_case'],
                expected_value=data['expected_value'],
                expected_return_pct=data['expected_return_pct'],
                risk_reward_ratio=abs(data['expected_return_pct'] / (data.get('max_loss', 1)))
            )
            
            return ReasoningOutput(
                ticker=ticker,
                recommended_contract=contract,
                scenario_analysis=scenario_analysis,
                chain_of_thought=data['chain_of_thought'],
                conviction_score=data['conviction_score'],
                break_even_price=data.get('break_even_price'),
                max_loss=data.get('max_loss'),
                projected_1yr_value=data['expected_value']
            )
            
        except Exception as e:
            print(f"⚠️ Error parsing reasoning response: {e}")
            # Return minimal output
            return ReasoningOutput(
                ticker=ticker,
                recommended_contract=contract,
                chain_of_thought=response,
                conviction_score=0.5
            )
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from response"""
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        return text.strip()


class RiskCritic(OpenRouterClient):
    """
    The "Safety" - plays devil's advocate using DeepSeek V3
    Identifies red flags and provides risk assessment
    """
    
    def __init__(self):
        super().__init__()
        self.model = config.openrouter.deepseek_v3_model
        self.leap_heuristics = config.leap_heuristics
    
    def critique_recommendation(self, ticker: str, reasoning: ReasoningOutput,
                               market_data: MarketData, earnings_date: Optional[datetime] = None) -> CritiqueOutput:
        """
        Perform critical risk assessment on a LEAP recommendation
        
        Args:
            ticker: Stock symbol
            reasoning: Output from QuantReasoningEngine
            market_data: Market data
            earnings_date: Next earnings date if available
            
        Returns:
            CritiqueOutput with risk flags and approval decision
        """
        prompt = self._build_critique_prompt(ticker, reasoning, market_data, earnings_date)
        
        messages = [
            {"role": "system", "content": "You are a risk management expert. Your job is to find problems and red flags in investment recommendations."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.call_llm(self.model, messages, temperature=0.2)
            
            critique_output = self._parse_critique_response(response, ticker, reasoning, market_data, earnings_date)
            
            print(f"✅ Risk critique completed for {ticker}")
            return critique_output
            
        except Exception as e:
            print(f"❌ Critique error for {ticker}: {e}")
            # Default to cautious stance
            return CritiqueOutput(
                ticker=ticker,
                approved=False,
                risk_flags=RiskFlags(critical_issues=["Error during critique"]),
                critique_reasoning=str(e),
                risk_score=1.0
            )
    
    def _build_critique_prompt(self, ticker: str, reasoning: ReasoningOutput,
                              market_data: MarketData, earnings_date: Optional[datetime]) -> str:
        """Build the critique prompt"""
        
        contract = reasoning.recommended_contract
        
        prompt = f"""# Risk Assessment: {ticker} LEAP Trade

## Proposed Trade
- **Ticker**: {ticker}
- **Strike**: ${contract.strike:.2f}
- **Expiration**: {contract.expiration.strftime('%Y-%m-%d')}
- **Entry Price**: ${((contract.bid + contract.ask) / 2):.2f}
- **Delta**: {contract.delta:.3f}
- **IV**: {contract.implied_volatility:.2%}
- **Bid/Ask Spread**: {contract.bid_ask_spread_pct:.2f}%

## Quant's Conviction
- **Score**: {reasoning.conviction_score}/1.0
- **Expected Return**: {reasoning.scenario_analysis.expected_return_pct if reasoning.scenario_analysis else 'N/A'}%
- **Break-Even**: ${reasoning.break_even_price:.2f}

## Market Context
- **IV Rank**: {market_data.implied_volatility_rank or 'Unknown'}
- **Next Earnings**: {earnings_date.strftime('%Y-%m-%d') if earnings_date else 'Unknown'}

## Your Task: Play Devil's Advocate
Identify RED FLAGS and potential issues with this trade:

1. **IV Risk**: Is IV too high? (Reject if IV > 40th percentile)
2. **Liquidity Risk**: Is spread too wide or volume too low?
3. **Theta Risk**: Will time decay be too aggressive?
4. **Earnings Risk**: Is earnings within the hold period?
5. **Leverage Risk**: Is the position too risky given market conditions?

## Provide Output in JSON:
```json
{{
  "approved": true/false,
  "risk_score": 0.0 to 1.0,
  "critique_reasoning": "Detailed explanation of concerns...",
  "risk_flags": {{
    "high_iv": true/false,
    "low_liquidity": true/false,
    "wide_spread": true/false,
    "upcoming_earnings": true/false,
    "insufficient_time_value": true/false
  }},
  "critical_issues": ["issue 1", "issue 2"],
  "warnings": ["warning 1"]
}}
```

Be harsh. Only approve if this is a genuinely sound trade.
"""
        return prompt
    
    def _parse_critique_response(self, response: str, ticker: str, reasoning: ReasoningOutput,
                                 market_data: MarketData, earnings_date: Optional[datetime]) -> CritiqueOutput:
        """Parse critique response"""
        try:
            json_str = response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            
            data = json.loads(json_str)
            
            # Build risk flags
            risk_flags_data = data.get('risk_flags', {})
            risk_flags = RiskFlags(
                high_iv=risk_flags_data.get('high_iv', False),
                low_liquidity=risk_flags_data.get('low_liquidity', False),
                wide_spread=risk_flags_data.get('wide_spread', False),
                upcoming_earnings=risk_flags_data.get('upcoming_earnings', False),
                insufficient_time_value=risk_flags_data.get('insufficient_time_value', False),
                critical_issues=data.get('critical_issues', []),
                warnings=data.get('warnings', [])
            )
            
            return CritiqueOutput(
                ticker=ticker,
                approved=data.get('approved', False),
                risk_flags=risk_flags,
                critique_reasoning=data.get('critique_reasoning', response),
                risk_score=data.get('risk_score', 0.5)
            )
            
        except Exception as e:
            print(f"⚠️ Error parsing critique: {e}")
            # Default to rejection
            return CritiqueOutput(
                ticker=ticker,
                approved=False,
                risk_flags=RiskFlags(critical_issues=["Parse error"]),
                critique_reasoning=response,
                risk_score=1.0
            )
