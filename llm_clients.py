"""
LLM Clients for LEAP Strategic Asset Engine

Modular client system for different LLM providers and models.
Keeps ai_agents.py focused on orchestrator-specific functionality.
"""
import os
import requests
import json
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def get_insights(self, ticker: str, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI insights for market analysis"""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if client is properly configured"""
        pass


class GrokClient(LLMClient):
    """Grok (xAI) client for market analysis"""
    
    def __init__(self):
        self.api_key = os.getenv('GROK_API_KEY')
        self.base_url = "https://api.x.ai/v1/chat/completions"
        self.model = "grok-beta"
    
    def get_insights(self, ticker: str, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get market insights using Grok"""
        
        if not self.api_key:
            return {"error": "No GROK_API_KEY found"}
        
        # Build prompt from market context
        prompt = self._build_prompt(ticker, market_context)
        
        # Call Grok API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result['choices'][0]['message']['content']
                
                # Parse JSON response
                try:
                    insights = json.loads(llm_output)
                    return insights
                except json.JSONDecodeError:
                    return {"error": "Failed to parse Grok response"}
            else:
                return {"error": f"Grok API failed: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Grok client error: {str(e)}"}
    
    def _build_prompt(self, ticker: str, context: Dict[str, Any]) -> str:
        """Build analysis prompt from market context"""
        
        return f"""
        Analyze {ticker} for LEAP options investment and provide insights:

        Current Market Context:
        - Stock Price: ${context.get('current_price', 0):.2f} (7d: {context.get('price_change_7d', 0):+.1f}%, 30d: {context.get('price_change_30d', 0):+.1f}%)
        - Market Regime: {context.get('market_regime', 'UNKNOWN')}
        - IV Percentile: {context.get('implied_volatility', 0):.1f}%
        - Available Contracts: {context.get('available_contracts', 0)}

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
    
    def is_configured(self) -> bool:
        """Check if Grok client is properly configured"""
        return bool(self.api_key)


class OpenAIClient(LLMClient):
    """OpenAI client for market analysis (future implementation)"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4"
    
    def get_insights(self, ticker: str, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get market insights using OpenAI (placeholder implementation)"""
        return {"error": "OpenAI client not yet implemented"}
    
    def is_configured(self) -> bool:
        """Check if OpenAI client is properly configured"""
        return bool(self.api_key)


class AnthropicClient(LLMClient):
    """Anthropic client for market analysis (future implementation)"""
    
    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-sonnet-20240229"
    
    def get_insights(self, ticker: str, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get market insights using Anthropic (placeholder implementation)"""
        return {"error": "Anthropic client not yet implemented"}
    
    def is_configured(self) -> bool:
        """Check if Anthropic client is properly configured"""
        return bool(self.api_key)


class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    _clients = {
        "grok": GrokClient,
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
    }
    
    @classmethod
    def create_client(cls, model_name: str) -> LLMClient:
        """Create LLM client by model name"""
        
        model_name = model_name.lower()
        
        if model_name not in cls._clients:
            available = ", ".join(cls._clients.keys())
            raise ValueError(f"Unsupported model: {model_name}. Available: {available}")
        
        return cls._clients[model_name]()
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available models"""
        return list(cls._clients.keys())


# Example usage:
# client = LLMClientFactory.create_client("grok")
# insights = client.get_insights("NVDA", market_context)
