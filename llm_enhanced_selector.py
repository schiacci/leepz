"""
LLM-Enhanced Option Selector for LEAP Strategic Asset Engine

Integrates with existing AI agents infrastructure for multi-LLM consensus 
dynamic scoring, sentiment analysis, and adaptive parameter selection.
Works alongside the existing quantitative OptionSelector.
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import os
import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import OptionContract, MarketData
from option_selector import OptionSelector as QuantOptionSelector
from quant_calculations import get_market_regime
from agents.ai_agents import LLMClient
from config import config


class LLMEnhancedOptionSelector:
    """
    LLM-Enhanced Option Selector that combines quantitative analysis with multi-LLM consensus
    
    Features:
    - Dynamic weight optimization based on market conditions
    - Real-time sentiment analysis via multiple LLM providers
    - Sequential LLM analysis with context building
    - Consensus-driven scoring and recommendations
    - Adaptive parameter selection
    - Narrative-driven explanations
    """
    
    def __init__(self, providers: List[str] = None, sentiment_model: str = None):
        self.quant_selector = QuantOptionSelector()
        self.sentiment_model = sentiment_model or "x-ai/grok-beta"  # Default to Grok
        
        # Default to all available providers if none specified
        if providers is None:
            self.providers = ["openrouter", "gemini"]  # Default providers
        else:
            self.providers = [p.lower() for p in providers]
        
        # Initialize LLM clients for each provider
        self.llm_clients = {}
        for provider in self.providers:
            try:
                # Create LLMClient with specific providers
                self.llm_clients[provider] = LLMClient(provider_ids=[provider])
                print(f"  🤖 Initialized LLM client: {provider.upper()}")
            except Exception as e:
                print(f"  ⚠️ Failed to initialize {provider}: {e}")
                continue
        
        if not self.llm_clients:
            print(f"  ⚠️ No LLM clients available, using baseline analysis only")
    
    def enhance_quantitative_results(
        self,
        ticker: str,
        market_data: MarketData,
        quantitative_result: Dict[str, any],
        max_iterations: int = 3,
        confidence_threshold: float = 0.85
    ) -> Dict[str, any]:
        """
        Enhance existing quantitative results with multi-LLM consensus
        
        Sequentially calls multiple LLMs, each building on previous insights.
        Stops early if confidence threshold is reached or max iterations exceeded.
        
        Args:
            ticker: Stock symbol
            market_data: Market data with option chain
            quantitative_result: Results from OptionSelector.recommend_optimal_structure()
            max_iterations: Maximum number of LLMs to consult (default: 3)
            confidence_threshold: Stop early if avg confidence exceeds this (default: 0.85)
        
        Returns:
            Enhanced results dictionary with multi-LLM consensus insights
        """
        try:
            print(f"  🤖 Multi-LLM Enhancement for {ticker}")
            
            if not quantitative_result.get('success'):
                return quantitative_result
            
            if not self.llm_clients:
                print(f"  ⚠️ No LLM clients available, using baseline analysis")
                return quantitative_result
            
            # Sequential LLM analysis with context building and early stopping
            all_insights = []
            accumulated_context = {
                'ticker': ticker,
                'quantitative_score': quantitative_result.get('score', 0),
                'quantitative_contract': quantitative_result.get('contract'),
                'previous_insights': []
            }
            
            # Limit to max_iterations providers
            providers_to_use = self.providers[:max_iterations]
            
            for i, provider in enumerate(providers_to_use):
                if provider not in self.llm_clients:
                    print(f"  ⚠️ Skipping unavailable provider: {provider}")
                    continue
                
                print(f"  🔄 LLM {i+1}/{len(providers_to_use)}: {provider.upper()}")
                
                # Get insights from current provider
                insight = self._get_llm_insights_with_context(
                    provider, ticker, market_data, accumulated_context
                )
                
                if insight:
                    insight['provider'] = provider
                    insight['sequence'] = i + 1
                    all_insights.append(insight)
                    accumulated_context['previous_insights'].append(insight)
                    
                    confidence = insight.get('confidence_level', 0)
                    print(f"  ✅ {provider.upper()} insights: sentiment {insight.get('sentiment_score', 0):.2f}, confidence {confidence:.2f}")
                    
                    # Check for early stopping if we have enough insights
                    if len(all_insights) >= 2:  # Need at least 2 for meaningful consensus
                        avg_confidence = sum(ins.get('confidence_level', 0) for ins in all_insights) / len(all_insights)
                        if avg_confidence >= confidence_threshold:
                            print(f"  🎯 High confidence ({avg_confidence:.2f} >= {confidence_threshold:.2f}) - stopping early")
                            break
                else:
                    print(f"  ❌ {provider.upper()} failed to provide insights")
            
            print(f"  📊 Consulted {len(all_insights)} LLMs (max: {max_iterations}, threshold: {confidence_threshold:.2f})")
            
            # Apply multi-LLM consensus enhancements
            enhanced_result = self._apply_multi_llm_consensus(
                quantitative_result, all_insights, market_data, ticker
            )
            
            return enhanced_result
            
        except Exception as e:
            print(f"  ⚠️ Multi-LLM enhancement failed: {e}")
            return quantitative_result
    
    def recommend_optimal_structure(
        self,
        ticker: str,
        market_data: MarketData,
        target_delta: float = 0.7,
        min_dte: int = 365,
        max_dte: int = 730,
        max_iv_percentile: float = 70.0,
        risk_tolerance: str = "MODERATE"
    ) -> Dict[str, any]:
        """
        Recommend optimal LEAP option structure with multi-LLM enhancement
        
        This method maintains backward compatibility and can be used independently.
        For new usage, prefer enhance_quantitative_results() method.
        
        Returns:
            Dictionary with enhanced recommendation details
        """
        try:
            print(f"  🤖 Multi-LLM Analysis for {ticker}")
            
            # Get quantitative baseline
            quant_result = self.quant_selector.recommend_optimal_structure(
                ticker, market_data, target_delta, min_dte, max_dte, max_iv_percentile, risk_tolerance
            )
            
            if not quant_result.get('success'):
                return quant_result
            
            # Use multi-LLM enhancement
            return self.enhance_quantitative_results(ticker, market_data, quant_result)
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Multi-LLM analysis failed: {str(e)}',
                'fallback': self.quant_selector.recommend_optimal_structure(
                    ticker, market_data, target_delta, min_dte, max_dte, max_iv_percentile, risk_tolerance
                )
            }
    
    def _get_llm_insights_with_context(
        self, provider: str, ticker: str, market_data: MarketData, 
        context: Dict[str, any]
    ) -> Dict[str, any]:
        """Get LLM insights with accumulated context from previous LLMs"""
        
        if provider not in self.llm_clients:
            return {}
        
        llm_client = self.llm_clients[provider]
        
        try:
            # Get current market regime
            regime = get_market_regime()
            
            # Get recent price data for context
            stock = yf.Ticker(ticker)
            hist = stock.history(period="30d")
            current_price = hist['Close'].iloc[-1]
            price_change_7d = ((current_price - hist['Close'].iloc[-8]) / hist['Close'].iloc[-8]) * 100
            price_change_30d = ((current_price - hist['Close'].iloc[-30]) / hist['Close'].iloc[-30]) * 100
            
            # Build context prompt with previous insights
            previous_insights_text = ""
            if context['previous_insights']:
                previous_insights_text = "\n".join([
                    f"LLM {insight['sequence']} ({insight['provider']}): "
                    f"Sentiment={insight.get('sentiment_score', 0):.2f}, "
                    f"Confidence={insight.get('confidence_level', 0):.2f}, "
                    f"Key Points: {insight.get('sentiment_reasoning', 'N/A')}"
                    for insight in context['previous_insights']
                ])
            
            # Enhanced prompt with context
            prompt = f"""
            Analyze {ticker} for LEAP options investment, building on previous AI insights:

            Current Market Context:
            - Stock Price: ${current_price:.2f} (7d: {price_change_7d:+.1f}%, 30d: {price_change_30d:+.1f}%)
            - Market Regime: {regime}
            - IV Percentile: {market_data.implied_volatility_rank:.1f}%
            - Available Contracts: {len(market_data.option_contracts)}
            - Quantitative Score: {context['quantitative_score']:.3f}

            Previous AI Insights:
            {previous_insights_text if previous_insights_text else "No previous insights - this is the initial analysis"}

            YOUR TASK:
            Provide your analysis considering all previous insights. Either:
            1. AGREE with previous consensus and add new insights
            2. DISAGREE and provide counter-reasoning
            3. REFINE previous analysis with additional context

            Provide analysis in JSON format:
            {{
                "sentiment_score": 0.0-1.0,
                "sentiment_reasoning": "your analysis and response to previous insights",
                "volatility_outlook": "LOW/MEDIUM/HIGH",
                "optimal_delta_adjustment": -0.1 to 0.1,
                "optimal_dte_adjustment": -180 to 180,
                "confidence_level": 0.0-1.0,
                "key_catalysts": ["catalyst1", "catalyst2"],
                "risk_factors": ["risk1", "risk2"],
                "consensus_position": "AGREE/DISAGREE/REFINE",
                "additional_insights": "any new perspectives you bring"
            }}

            Focus on:
            - Evaluating previous AI insights critically
            - Adding your unique perspective on this stock
            - Technical analysis and market sentiment
            - Catalysts and risks others may have missed
            """
            
            # Get appropriate model for provider
            # For OpenRouter, use the specified sentiment model
            if provider == "openrouter":
                model = self.sentiment_model  # Use user-specified sentiment model
            elif provider == "gemini":
                model = config.google_ai.model
            else:
                model = "fallback-model"
            
            # Call LLM with context
            messages = [{"role": "user", "content": prompt}]
            
            # Check if this is an Ollama provider to enable streaming
            is_ollama = "ollama" in provider.lower()
            
            # Create a simple stream callback for Ollama models
            def stream_callback(chunk):
                if chunk.strip():
                    print(chunk, end='', flush=True)
            
            llm_response = llm_client.call_llm(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000,  # Increased from 600 to get complete JSON
                stream=is_ollama,  # Enable streaming for Ollama models
                stream_callback=stream_callback if is_ollama else None
            )
            
            # Parse JSON from LLM response
            try:
                # If llm_response is already a dict (from Ollama), use it directly
                if isinstance(llm_response, dict):
                    insights = llm_response
                else:
                    # Otherwise parse as JSON string
                    insights = json.loads(llm_response)
                print(f"  ✅ {provider.upper()} insights: sentiment {insights.get('sentiment_score', 0.5):.2f}, confidence {insights.get('confidence_level', 0.5):.2f}")
                return insights
            except (json.JSONDecodeError, TypeError) as e:
                print(f"  ⚠️ {provider.upper()} response parsing failed: {e}")
                return {}
                
        except Exception as e:
            print(f"  ⚠️ {provider.upper()} analysis error: {e}")
            return {}
    
    def _apply_multi_llm_consensus(
        self, 
        quant_result: Dict[str, any], 
        all_insights: List[Dict[str, any]],
        market_data: MarketData,
        ticker: str
    ) -> Dict[str, any]:
        """Apply multi-LLM consensus to enhance quantitative recommendation"""
        
        if not all_insights:
            return quant_result
        
        # Calculate consensus metrics
        sentiment_scores = [insight.get('sentiment_score', 0.5) for insight in all_insights]
        confidence_scores = [insight.get('confidence_level', 0.5) for insight in all_insights]
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Weight scores by confidence
        weighted_sentiment = sum(s * c for s, c in zip(sentiment_scores, confidence_scores)) / sum(confidence_scores)
        
        # Apply consensus enhancement to quantitative score
        base_score = quant_result.get('score', 0.744)
        
        # Sentiment adjustment (consensus-driven)
        if weighted_sentiment > 0.6:  # Bullish consensus
            sentiment_boost = 0.05 * weighted_sentiment
        elif weighted_sentiment < 0.4:  # Bearish consensus
            sentiment_boost = -0.05 * (1 - weighted_sentiment)
        else:  # Neutral consensus
            sentiment_boost = 0.0
        
        # Confidence adjustment
        confidence_multiplier = 0.8 + (0.4 * avg_confidence)  # 0.8 to 1.2 range
        
        enhanced_score = base_score * confidence_multiplier + sentiment_boost
        enhanced_score = min(max(enhanced_score, 0.0), 1.0)  # Clamp to [0,1]
        
        # Build consensus narrative
        consensus_positions = [insight.get('consensus_position', 'AGREE') for insight in all_insights]
        agreement_count = consensus_positions.count('AGREE')
        consensus_strength = agreement_count / len(consensus_positions)
        
        # Compile all catalysts and risks
        all_catalysts = []
        all_risks = []
        for insight in all_insights:
            all_catalysts.extend(insight.get('key_catalysts', []))
            all_risks.extend(insight.get('risk_factors', []))
        
        # Create enhanced result
        enhanced_result = quant_result.copy()
        enhanced_result.update({
            'original_score': base_score,
            'enhanced_score': enhanced_score,
            'confidence_factor': confidence_multiplier,
            'consensus_sentiment': weighted_sentiment,
            'avg_confidence': avg_confidence,
            'consensus_strength': consensus_strength,
            'llm_insights': all_insights,
            'all_catalysts': list(set(all_catalysts)),  # Remove duplicates
            'all_risks': list(set(all_risks)),
            'multi_llm_analysis': {
                'providers_used': [insight['provider'] for insight in all_insights],
                'sentiment_range': [min(sentiment_scores), max(sentiment_scores)],
                'confidence_range': [min(confidence_scores), max(confidence_scores)],
                'agreement_rate': consensus_strength
            }
        })
        
        print(f"  🤖 Multi-LLM Consensus: {len(all_insights)} providers")
        print(f"  📊 Consensus Sentiment: {weighted_sentiment:.2f}")
        print(f"  🎯 Enhanced Score: {enhanced_score:.3f} (Original: {base_score:.3f})")
        
        return enhanced_result
