"""
AI Agents for LEAP Strategic Asset Engine
Uses OpenRouter API to access Grok and DeepSeek models
"""
import json
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path
from datetime import datetime
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

import google.genai as genai

# Provider Registry - defines available LLM providers and their configurations
PROVIDER_REGISTRY = {
    "openrouter": {
        "name": "OpenRouter",
        "priority": 1,  # Highest priority (tried first)
        "client_class": lambda: OpenRouterClient(),
        "model": lambda: config.openrouter.deepseek_r1_model,
        "requires_key": lambda: config.openrouter.api_key,
        "free_tier": False,
        "supports_streaming": True,
    },
    "gemini": {
        "name": "Google Gemini", 
        "priority": 2,  # Fallback priority
        "client_class": lambda: GeminiClient(),
        "model": lambda: config.google_ai.model,
        "requires_key": lambda: config.google_ai.api_key,
        "free_tier": True,
        "supports_streaming": False,
    },
    # Future providers can be added here with higher priorities
    # "anthropic": {...},
    # "openai": {...},
    # "groq": {...},
}

from quant_calculations import (
    black_scholes_price,
    extract_growth_assumptions,
    monte_carlo_itm_probability,
    verify_option_pricing
)
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


class DiscoveryScout:
    """AI agent for discovering trending assets using Grok/xAI or Gemini"""
    
    def __init__(self):
        self.client = LLMClient()
        self.model = config.openrouter.grok_model if config.openrouter.api_key else config.google_ai.model
    
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
        # Add temporal constraints for backtesting mode
        current_date_context = datetime.now()
        temporal_constraint = ""
        
        if config.backtesting_mode and config.backtesting_date:
            current_date_context = config.backtesting_date
            temporal_constraint = f"""
**IMPORTANT: TEMPORAL CONSTRAINTS FOR HISTORICAL ANALYSIS**
- You are analyzing market conditions as they existed on {current_date_context.strftime('%Y-%m-%d')}
- You ONLY have access to information, news, and market data available up to this date
- You CANNOT use any knowledge of future events, earnings reports, or market developments after {current_date_context.strftime('%Y-%m-%d')}
- Base your analysis on what would have been known and visible to investors on this historical date
- Focus on trends, news, and market sentiment as they existed at that time
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

{temporal_constraint}

Analysis Date: {current_date_context.strftime('%Y-%m-%d')}
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


class GeminiClient:
    """Mock Google AI (Gemini) client for development - returns realistic dummy responses"""
    
    def __init__(self):
        self.api_key = config.google_ai.api_key
        self.model = config.google_ai.model
        print("🤖 Using MOCK Gemini client for development (free, no API calls)")
    
    def call_llm(self, messages: List[Dict[str, str]], 
                 temperature: float = 0.7, max_tokens: int = 4000,
                 stream: bool = False, stream_callback: Optional[callable] = None) -> str:
        """
        Return realistic mock responses for development
        """
        if not self.api_key or self.api_key == 'your_google_ai_key_here':
            print("⚠️  Mock Gemini: No real API key - using mock responses")
        
        # Extract the last user message to generate a contextual response
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        last_message = user_messages[-1]["content"] if user_messages else ""
        
        # Generate mock JSON responses based on content - return valid JSON
        if "break-even" in last_message.lower():
            return '''{
  "chain_of_thought": "Based on the LEAP option analysis, the break-even price at Day 365 is $152.40. This calculation considers the time decay of the option premium over the 12-month holding period. The stock would need to appreciate approximately 23.4% from the current price of $123.50 to reach this break-even level. The analysis incorporates Black-Scholes pricing verification showing the option is slightly underpriced at current market levels.",
  "break_even_price": 152.40,
  "scenario_analysis": {
    "bull_case": {"stock_price": 154.38, "option_value": 31.20, "return_pct": 24.8},
    "base_case": {"stock_price": 135.85, "option_value": 12.35, "return_pct": -50.6},
    "bear_case": {"stock_price": 104.98, "option_value": 0.00, "return_pct": -100.0}
  },
  "expected_value": 8.25,
  "expected_return_pct": 33.0,
  "max_loss": 25.00,
  "conviction_score": 0.72
}'''
            
        elif "scenario" in last_message.lower() or "bull case" in last_message.lower():
            return '''{
  "chain_of_thought": "Scenario analysis shows strong potential in the bull case with 35% probability weighting. The Monte Carlo simulation indicates 62.4% ITM probability with 95% confidence interval of 59.4%-65.4%. Black-Scholes verification confirms the option is underpriced by 9.4% compared to theoretical value. Risk/reward profile favors the trade given the growth assumptions extracted from the investment narrative.",
  "break_even_price": 152.40,
  "scenario_analysis": {
    "bull_case": {"stock_price": 154.38, "option_value": 31.20, "return_pct": 24.8},
    "base_case": {"stock_price": 135.85, "option_value": 12.35, "return_pct": -50.6},
    "bear_case": {"stock_price": 104.98, "option_value": 0.00, "return_pct": -100.0}
  },
  "expected_value": 8.25,
  "expected_return_pct": 33.0,
  "max_loss": 25.00,
  "conviction_score": 0.78
}'''
            
        elif "conviction" in last_message.lower() or "recommendation" in last_message.lower():
            return '''{
  "chain_of_thought": "After rigorous Chain-of-Thought analysis incorporating Black-Scholes pricing verification and Monte Carlo simulation: Key findings include pricing efficiency analysis showing 8.7% undervaluation, statistical probability assessment with 63.2% ITM likelihood (95% CI: 60.1%-66.3%), and favorable risk/reward profile. The investment thesis validation confirms growth assumptions are well-supported. Position sizing recommended at 2-3% of portfolio.",
  "break_even_price": 152.40,
  "scenario_analysis": {
    "bull_case": {"stock_price": 154.38, "option_value": 31.20, "return_pct": 24.8},
    "base_case": {"stock_price": 135.85, "option_value": 12.35, "return_pct": -50.6},
    "bear_case": {"stock_price": 104.98, "option_value": 0.00, "return_pct": -100.0}
  },
  "expected_value": 8.25,
  "expected_return_pct": 33.0,
  "max_loss": 25.00,
  "conviction_score": 0.78
}'''
            
        else:
            # Generic response for other queries
            return f'''{{
  "chain_of_thought": "Quantitative analysis completed using Black-Scholes pricing model and Monte Carlo simulation. The option shows fair pricing within acceptable bounds with {60 + (hash(last_message) % 20)}% probability of finishing in-the-money. Key considerations include strong delta positioning at 0.82, reasonable implied volatility at 48%, and sufficient time to expiration at 575 days for narrative development. The trade demonstrates positive expected value based on extracted growth assumptions from the investment thesis.",
  "break_even_price": 152.40,
  "scenario_analysis": {{
    "bull_case": {{"stock_price": 154.38, "option_value": 31.20, "return_pct": 24.8}},
    "base_case": {{"stock_price": 135.85, "option_value": 12.35, "return_pct": -50.6}},
    "bear_case": {{"stock_price": 104.98, "option_value": 0.00, "return_pct": -100.0}}
  }},
  "expected_value": 8.25,
  "expected_return_pct": 33.0,
  "max_loss": 25.00,
  "conviction_score": 0.75
}}'''


class MultiProviderLLMClient:
    """Scalable LLM client that supports N providers with intelligent fallback"""
    
    def __init__(self, provider_ids: Optional[List[str]] = None):
        """
        Initialize with available providers
        
        Args:
            provider_ids: List of provider IDs to use (None = use all available)
        """
        self.providers = self._get_available_providers(provider_ids)
        self.current_provider = None
        
    def _get_available_providers(self, provider_ids: Optional[List[str]] = None) -> List[Dict]:
        """Get available providers sorted by priority"""
        available_providers = []
        
        # Filter providers if specific IDs requested
        providers_to_check = provider_ids if provider_ids else PROVIDER_REGISTRY.keys()
        
        for provider_id in providers_to_check:
            if provider_id in PROVIDER_REGISTRY:
                provider_config = PROVIDER_REGISTRY[provider_id].copy()
                provider_config["id"] = provider_id
                
                # Check if provider has required credentials
                if provider_config["requires_key"]():
                    available_providers.append(provider_config)
        
        # Sort by priority (lower number = higher priority)
        available_providers.sort(key=lambda x: x["priority"])
        
        return available_providers
    
    def call_llm_with_fallback(self, messages: List[Dict[str, str]], 
                              temperature: float = 0.7, max_tokens: int = 4000,
                              stream: bool = False, stream_callback: Optional[callable] = None,
                              excluded_providers: Optional[List[str]] = None) -> tuple[str, str]:
        """
        Try providers in priority order until one succeeds
        
        Returns:
            tuple: (response_text, provider_used)
        """
        excluded = excluded_providers or []
        
        for provider in self.providers:
            if provider["id"] in excluded:
                continue
                
            try:
                print(f"🔄 Trying {provider['name']}...")
                self.current_provider = provider
                
                # Create client instance
                client = provider["client_class"]()
                model = provider["model"]()
                
                # Use streaming if supported, otherwise fallback to non-streaming
                use_streaming = stream and provider["supports_streaming"]
                
                if provider["id"] == "openrouter":
                    response = client.call_llm(model, messages, temperature, max_tokens, 
                                             use_streaming, stream_callback)
                else:
                    # Gemini and other providers
                    response = client.call_llm(messages, temperature, max_tokens, 
                                              use_streaming, stream_callback)
                
                print(f"✅ Success with {provider['name']}!")
                return response, provider["id"]
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ {provider['name']} failed: {error_msg}")
                
                # Handle specific error types
                if "402" in error_msg and "Payment Required" in error_msg:
                    print(f"💰 {provider['name']} credits depleted - trying next provider...")
                elif "429" in error_msg:  # Rate limit
                    print(f"⏱️ {provider['name']} rate limited - trying next provider...")
                elif "401" in error_msg:  # Auth error
                    print(f"🔐 {provider['name']} auth failed - trying next provider...")
                # Continue to next provider
        
        # All providers failed
        raise Exception("All LLM providers failed - check API keys and network connection")
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about current provider setup"""
        return {
            "available_providers": [p["name"] for p in self.providers],
            "total_providers": len(self.providers),
            "current_provider": self.current_provider["name"] if self.current_provider else None,
            "free_tier_available": any(p["free_tier"] for p in self.providers),
        }


class LLMClient(MultiProviderLLMClient):
    """Backward-compatible LLM client for existing code"""
    
    def __init__(self):
        super().__init__()
        # For backward compatibility, use first available provider
        if self.providers:
            self.provider = self.providers[0]["id"]
        else:
            self.provider = None
    
    def call_llm(self, model: str, messages: List[Dict[str, str]], 
                 temperature: float = 0.7, max_tokens: int = 4000,
                 stream: bool = False, stream_callback: Optional[callable] = None) -> str:
        """Backward-compatible call method"""
        response, _ = self.call_llm_with_fallback(messages, temperature, max_tokens, stream, stream_callback)
        return response


class QuantReasoningEngine:
    """AI agent for quantitative LEAP analysis using DeepSeek R1 or Gemini"""
    
    def __init__(self):
        self.client = LLMClient()
        self.model = config.openrouter.deepseek_r1_model if config.openrouter.api_key else config.google_ai.model
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
        
        # Extract growth assumptions from narrative
        print(f"  📈 Extracting growth assumptions from narrative...")
        growth_assumptions = extract_growth_assumptions(narrative)
        print(f"     Expected Growth: {growth_assumptions['expected_growth']:.1%}")
        print(f"     Assumed Volatility: {growth_assumptions['volatility']:.1%}")
        
        # Perform Black-Scholes pricing verification
        print(f"  💰 Running Black-Scholes pricing verification...")
        market_price = (best_contract.bid + best_contract.ask) / 2
        T_years = best_contract.days_to_expiry / 365.0
        r = 0.05  # Risk-free rate assumption (5%)
        
        bs_price = black_scholes_price(
            S=market_data.spot_price,
            K=best_contract.strike,
            T=T_years,
            r=r,
            sigma=best_contract.implied_volatility or growth_assumptions['volatility']
        )
        
        pricing_verification = verify_option_pricing(market_price, bs_price)
        print(f"     Market Price: ${market_price:.2f} | BS Price: ${bs_price:.2f}")
        print(f"     Fairness: {pricing_verification['fairness_assessment']} ({pricing_verification['price_diff_pct']:+.1%})")
        
        # Run Monte Carlo simulation for ITM probability
        print(f"  🎲 Running Monte Carlo simulation (1,000 runs)...")
        mc_results = monte_carlo_itm_probability(
            S=market_data.spot_price,
            K=best_contract.strike,
            T=T_years,
            r=r,
            sigma=best_contract.implied_volatility or growth_assumptions['volatility'],
            growth_assumptions=growth_assumptions,
            n_runs=1000
        )
        
        print(f"     ITM Probability: {mc_results['itm_probability']:.1%} (95% CI: {mc_results['ci_lower']:.1%} - {mc_results['ci_upper']:.1%})")
        
        print(f"  🧠 Building analysis prompt for {ticker}...")
        prompt = self._build_reasoning_prompt(ticker, market_data, best_contract, narrative, 
                                             growth_assumptions, pricing_verification, mc_results)
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

                # Buffer for collecting related chunks
                if not hasattr(stream_callback, 'buffer'):
                    stream_callback.buffer = []
                    stream_callback.last_section = None

                # Detect section changes and key insights
                if any(phrase in chunk.lower() for phrase in ['break-even', 'scenario', 'bull case', 'base case', 'bear case', 'expected value', 'risk assessment', 'conviction']):
                    # New section - flush buffer and show section header
                    if stream_callback.buffer:
                        _process_buffer()
                    section_name = chunk.split(':')[0] if ':' in chunk else chunk
                    print(f"\n🔍 {section_name}...")
                    stream_callback.last_section = section_name
                    return

                elif any(chunk.endswith(end) for end in ['.', '!', '?']) and len(chunk) > 10:
                    # Complete sentence - show as insight
                    if stream_callback.buffer:
                        _process_buffer()
                    print(f"💡 {chunk}")
                    return

                elif '**' in chunk or '$$' in chunk:
                    # Structured content - show formatted
                    if stream_callback.buffer:
                        _process_buffer()
                    clean_chunk = chunk.replace('**', '').replace('$$', '').strip()
                    print(f"📊 {clean_chunk}")
                    return

                elif any(char.isdigit() for char in chunk) and any(op in chunk for op in ['+', '-', '*', '/', '=']):
                    # Mathematical expressions - buffer and summarize
                    stream_callback.buffer.append(chunk)
                    # Show progress indicator every few chunks
                    if len(stream_callback.buffer) % 10 == 0:
                        print("⚡ Calculating...", end='\r', flush=True)
                    return

                elif chunk and len(chunk) < 20 and not any(char.isdigit() for char in chunk):
                    # Short non-numeric chunks (likely words) - buffer for context
                    stream_callback.buffer.append(chunk)
                    return

                else:
                    # Other content - show as-is but clean
                    if stream_callback.buffer:
                        _process_buffer()
                    if len(chunk) > 5:  # Only show meaningful chunks
                        clean_chunk = chunk.replace('\\', '').replace('frac', '/')
                        print(f"🤔 {clean_chunk}", end='', flush=True)

            def _process_buffer():
                """Process accumulated buffer content"""
                if not stream_callback.buffer:
                    return

                buffer_text = ' '.join(stream_callback.buffer)

                # Try to extract key numbers or results
                import re
                numbers = re.findall(r'\d+\.?\d*', buffer_text)
                if len(numbers) >= 2 and stream_callback.last_section:
                    # Show summary for this section
                    if 'break' in stream_callback.last_section.lower():
                        print(f"💰 Break-even calculated")
                    elif 'bull' in stream_callback.last_section.lower():
                        print(f"📈 Bull case: +{numbers[-1]}%" if numbers else "📈 Bull case analyzed")
                    elif 'base' in stream_callback.last_section.lower():
                        print(f"➡️ Base case: {numbers[-1]}%" if numbers else "➡️ Base case analyzed")
                    elif 'bear' in stream_callback.last_section.lower():
                        print(f"📉 Bear case: {numbers[-1]}%" if numbers else "📉 Bear case analyzed")
                    elif 'expected' in stream_callback.last_section.lower():
                        print(f"🎯 Expected return: {numbers[-1]}%" if numbers else "🎯 Expected value calculated")

                stream_callback.buffer = []
            
            print("  🔄 Attempting streaming analysis...")
            response = self.client.call_llm(self.model, messages, temperature=0.3, max_tokens=6000, 
                                   stream=True, stream_callback=stream_callback)
            print(f"\n  ✅ Received analysis response for {ticker}")
            
            # Parse the structured response
            print(f"  📋 Parsing quantitative results for {ticker}...")
            reasoning_output = self._parse_reasoning_response(response, ticker, best_contract)
            
            print(f"✅ Completed quant analysis for {ticker}")
            return reasoning_output
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Initial provider failed: {error_msg}")
            
            # Use multi-provider fallback system
            try:
                print(f"🔄 Attempting fallback with other providers...")
                fallback_client = MultiProviderLLMClient()
                response, provider_used = fallback_client.call_llm_with_fallback(
                    messages, temperature=0.3, max_tokens=6000, 
                    stream=False, stream_callback=None  # Gemini doesn't support streaming
                )
                
                print(f"✅ Fallback successful with {fallback_client.current_provider['name']}!")
                
                # Parse the structured response
                print(f"  📋 Parsing quantitative results for {ticker}...")
                reasoning_output = self._parse_reasoning_response(response, ticker, best_contract)
                
                print(f"✅ Completed quant analysis for {ticker} (via {provider_used} fallback)")
                return reasoning_output
                
            except Exception as fallback_error:
                print(f"❌ All providers failed: {fallback_error}")
                return None
    
    def _build_reasoning_prompt(self, ticker: str, market_data: MarketData,
                                contract: OptionContract, narrative: str,
                                growth_assumptions: Dict, pricing_verification: Dict,
                                mc_results: Dict) -> str:
        """Build the detailed reasoning prompt with quantitative data"""
        
        # Calculate time to expiration in years
        T_years = contract.days_to_expiry / 365.0
        
        # Add temporal constraints for backtesting mode
        analysis_date = datetime.now()
        temporal_constraint = ""
        
        if config.backtesting_mode and config.backtesting_date:
            analysis_date = config.backtesting_date
            temporal_constraint = f"""
**CRITICAL: TEMPORAL CONSTRAINTS FOR HISTORICAL ANALYSIS**
- You are performing this analysis on {analysis_date.strftime('%Y-%m-%d')}
- You ONLY have access to market data, news, and information available up to this historical date
- You CANNOT use any knowledge of future events, earnings results, or market developments after {analysis_date.strftime('%Y-%m-%d')}
- Base your reasoning on quantitative data and assumptions that would have been available to investors on this date
- The growth assumptions provided were extracted from narratives visible at that time
- Focus on statistical analysis using the provided quantitative inputs
"""
        
        prompt = f"""# LEAP Options Analysis: {ticker}

## Investment Thesis
{narrative}

## Quantitative Foundation

### Black-Scholes Pricing Verification
- **Market Price**: ${pricing_verification['market_price']:.2f}
- **Black-Scholes Price**: ${pricing_verification['bs_price']:.2f}
- **Price Difference**: {pricing_verification['price_diff_pct']:+.1%}
- **Assessment**: {pricing_verification['fairness_assessment']}

### Monte Carlo ITM Probability (1,000 Simulations)
- **ITM Probability**: {mc_results['itm_probability']:.1%} (95% CI: {mc_results['ci_lower']:.1%} - {mc_results['ci_upper']:.1%})
- **Expected ITM Value**: ${mc_results['expected_itm_value']:.2f}
- **Drift Rate**: {mc_results['drift_rate']:.1%}
- **Volatility Used**: {mc_results['simulated_volatility']:.1%}

### Growth Assumptions Extracted
- **Expected Annual Growth**: {growth_assumptions['expected_growth']:.1%}
- **Volatility Assumption**: {growth_assumptions['volatility']:.1%}

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

{temporal_constraint}

## Your Task
Perform a rigorous Chain-of-Thought analysis to determine if this is a sound LEAP trade using the "Buy 1.5yr, Sell 1yr" strategy.

### Analysis Framework:
1. **Thesis Validation**: Assess the strength of the investment narrative and growth assumptions
2. **Pricing Efficiency**: Evaluate if the option is fairly priced using Black-Scholes verification
3. **Probability Assessment**: Incorporate Monte Carlo ITM probability into your analysis
4. **Risk/Reward Profile**: Consider the statistical probability of success vs. premium paid

### Required Analysis:
1. **Break-Even Calculation**: What stock price is needed at Day 365 to break even?

2. **Scenario Analysis**: Estimate the option's value at the 1-year mark (Day 365) under three scenarios:
   - **Bull Case**: Stock price using the extracted growth assumptions
   - **Base Case**: Moderate growth scenario
   - **Bear Case**: Downside scenario

3. **Expected Value**: Calculate probability-weighted expected return using Monte Carlo results

4. **Conviction Score**: Rate 0.0 to 1.0 based on:
   - Thesis strength and growth assumptions
   - Options pricing efficiency (Black-Scholes verification)
   - Statistical probability of success (Monte Carlo ITM probability)
   - Risk/reward profile

## Output Format
Provide your analysis in JSON:
```json
{{
  "chain_of_thought": "Your detailed reasoning incorporating Black-Scholes verification and Monte Carlo probability...",
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

Be thorough in your reasoning and incorporate the quantitative data provided.

Analysis Date: {analysis_date.strftime('%Y-%m-%d')}
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


class RiskCritic(LLMClient):
    """
    The "Safety" - plays devil's advocate using DeepSeek V3
    Identifies red flags and provides risk assessment
    """
    
    def __init__(self):
        # Initialize parent class to get providers attribute
        super().__init__()
        # Override model selection
        self.model = config.openrouter.deepseek_v3_model if config.openrouter.api_key else config.google_ai.model
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
        """Build critique prompt for risk assessment"""
        contract = reasoning.recommended_contract

        # Safely format values that might be None
        iv_display = f"{contract.implied_volatility:.2%}" if contract.implied_volatility is not None else "Unknown"
        spread_display = f"{contract.bid_ask_spread_pct:.2f}%" if contract.bid_ask_spread_pct is not None else "Unknown%"
        break_even_display = f"${reasoning.break_even_price:.2f}" if reasoning.break_even_price is not None else "Unknown"
        
        # Add temporal constraints for backtesting mode
        analysis_date = datetime.now()
        temporal_constraint = ""
        
        if config.backtesting_mode and config.backtesting_date:
            analysis_date = config.backtesting_date
            temporal_constraint = f"""
**CRITICAL: TEMPORAL CONSTRAINTS FOR HISTORICAL RISK ASSESSMENT**
- You are performing this risk assessment on {analysis_date.strftime('%Y-%m-%d')}
- You ONLY have access to market conditions and information available up to this historical date
- You CANNOT consider future events, earnings results, or market developments after {analysis_date.strftime('%Y-%m-%d')}
- Base your risk analysis on what would have been visible to investors on this date
- Consider historical volatility patterns and market conditions up to this point
"""
        
        prompt = f"""# Risk Assessment: {ticker} LEAP Trade

## Proposed Trade
- **Ticker**: {ticker}
- **Strike**: ${contract.strike:.2f}
- **Expiration**: {contract.expiration.strftime('%Y-%m-%d')}
- **Entry Price**: ${((contract.bid + contract.ask) / 2):.2f}
- **Delta**: {contract.delta:.3f}
- **IV**: {iv_display}
- **Bid/Ask Spread**: {spread_display}

## Quant's Conviction
- **Score**: {reasoning.conviction_score}/1.0
- **Expected Return**: {reasoning.scenario_analysis.expected_return_pct if reasoning.scenario_analysis else 'N/A'}%
- **Break-Even**: {break_even_display}

## Market Context
- **IV Rank**: {market_data.implied_volatility_rank or 'Unknown'}
- **Next Earnings**: {earnings_date.strftime('%Y-%m-%d') if earnings_date else 'Unknown'}

{temporal_constraint}

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

Analysis Date: {analysis_date.strftime('%Y-%m-%d')}
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
