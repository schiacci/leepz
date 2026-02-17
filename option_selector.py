"""
Option Structure Recommendation Engine for LEAP Strategic Asset Engine

Recommends optimal LEAP option structures based on:
- Target delta (default 0.7)
- Expiration range (12-24 months)
- Max IV percentile
- Risk tolerance and market conditions
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from models import OptionContract, MarketData
from quant_calculations import black_scholes_price, black_scholes_greeks
from config import config


class OptionSelector:
    """
    LEAP Option Structure Recommendation Engine
    
    Selects optimal option contracts based on multiple criteria:
    - Target delta (0.7 default for deep ITM LEAPs)
    - Expiration timing (12-24 months)
    - IV percentile constraints
    - Liquidity and spread requirements
    """
    
    def __init__(self):
        self.heuristics = config.leap_heuristics
    
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
        Recommend optimal LEAP option structure
        
        Args:
            ticker: Stock symbol
            market_data: Market data with option chain
            target_delta: Target delta (0.7 default for LEAPs)
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            max_iv_percentile: Maximum IV percentile allowed
            risk_tolerance: CONSERVATIVE, MODERATE, or AGGRESSIVE
        
        Returns:
            Dictionary with recommendation details
        """
        try:
            # Debug: Check what we received
            print(f"  🔍 OptionSelector received {len(market_data.option_contracts)} contracts")
            
            # Filter contracts based on criteria
            filtered_contracts = self._filter_contracts(
                market_data.option_contracts,
                target_delta,
                min_dte,
                max_dte,
                max_iv_percentile
            )
            
            print(f"  🔍 OptionSelector filtered to {len(filtered_contracts)} contracts")
            
            if not filtered_contracts:
                return {
                    'success': False,
                    'message': 'No suitable contracts found',
                    'suggestions': self._generate_suggestions(market_data),
                    'debug_info': {
                        'total_contracts': len(market_data.option_contracts),
                        'target_delta': target_delta,
                        'min_dte': min_dte,
                        'max_dte': max_dte,
                        'max_iv_percentile': max_iv_percentile
                    }
                }
            
            # Score and rank contracts
            scored_contracts = self._score_contracts(
                filtered_contracts,
                market_data,
                target_delta,
                risk_tolerance
            )
            
            # Get best recommendation
            best_contract, best_score = scored_contracts[0]  # Unpack the tuple
            
            # Calculate additional metrics
            analysis = self._analyze_contract(best_contract, market_data)
            
            return {
                'success': True,
                'recommendation': {
                    'contract': best_contract,
                    'analysis': analysis,
                    'risk_assessment': self._assess_risk(best_contract, market_data),
                    'score': best_score
                },
                'alternative_contracts': scored_contracts[1:4],  # Top 3 alternatives
                'selection_criteria': {
                    'target_delta': target_delta,
                    'min_dte': min_dte,
                    'max_dte': max_dte,
                    'max_iv_percentile': max_iv_percentile,
                    'risk_tolerance': risk_tolerance
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error recommending structure: {str(e)}'
            }
    
    def _filter_contracts(
        self,
        contracts: List[OptionContract],
        target_delta: float,
        min_dte: int,
        max_dte: int,
        max_iv_percentile: float
    ) -> List[OptionContract]:
        """Filter contracts based on basic criteria"""
        print(f"  🔍 _filter_contracts: Starting with {len(contracts)} contracts")
        print(f"  🔍 Criteria: delta={target_delta}, DTE={min_dte}-{max_dte}, IV≤{max_iv_percentile}%")
        
        filtered = []
        
        for i, contract in enumerate(contracts):
            # Days to expiration filter - more flexible for LEAPs
            if not (min_dte <= contract.days_to_expiry <= max_dte):
                print(f"    ❌ Contract {i}: DTE {contract.days_to_expiry} outside range {min_dte}-{max_dte}")
                continue
            
            # Delta range (allow flexibility around target)
            delta_range = 0.2 if target_delta >= 0.7 else 0.3
            if not (target_delta - delta_range <= contract.delta <= target_delta + delta_range):
                print(f"    ❌ Contract {i}: Delta {contract.delta} outside target {target_delta}±{delta_range}")
                continue
            
            # IV percentile filter
            if contract.implied_volatility:
                # Convert IV to percentile estimate (simplified)
                # implied_volatility is already a decimal (0.72 = 72%), so just use it
                iv_percentile = contract.implied_volatility * 100
                
                if iv_percentile > max_iv_percentile:
                    print(f"    ❌ Contract {i}: IV {iv_percentile:.1f}% > max {max_iv_percentile}%")
                    continue
            
            # Liquidity filter
            if contract.bid_ask_spread_pct > self.heuristics.max_bid_ask_spread_pct:
                print(f"    ❌ Contract {i}: Spread {contract.bid_ask_spread_pct:.1f}% > max {self.heuristics.max_bid_ask_spread_pct}%")
                continue
            
            # Only ITM or slightly ITM for LEAPs
            # Allow more flexibility for LEAP recommendations
            if not contract.in_the_money:
                # Allow slightly OTM if very close to ATM (within 5%)
                if contract.strike > contract.last_price * 1.05:
                    print(f"    ❌ Contract {i}: OTM strike {contract.strike} > 105% of price {contract.last_price}")
                    continue
                else:
                    print(f"    ⚠️ Contract {i}: OTM but close to ATM (strike {contract.strike} <= 105% of price {contract.last_price})")
            
            print(f"    ✅ Contract {i}: PASSED all filters")
            filtered.append(contract)
        
        print(f"  🔍 _filter_contracts: {len(filtered)} contracts passed filtering")
        return filtered
    
    def _score_contracts(
        self,
        contracts: List[OptionContract],
        market_data: MarketData,
        target_delta: float,
        risk_tolerance: str
    ) -> List[Tuple[OptionContract, float]]:
        """Score contracts based on comprehensive weighted model"""
        from quant_calculations import get_market_regime
        
        scored = []
        current_regime = get_market_regime()
        
        print(f"  🔍 Scoring {len(contracts)} contracts with regime: {current_regime}")
        
        for contract in contracts:
            score = 0.0
            
            # 1. Delta proximity (25% weight) - Most important for LEAPs
            delta_score = 1.0 - abs(contract.delta - target_delta)
            score += delta_score * 0.25
            
            # 2. Regime-aware expiration scoring (25% weight)
            dte_score = self._calculate_regime_aware_dte_score(
                contract.days_to_expiry, current_regime
            )
            score += dte_score * 0.25
            
            # 3. Liquidity (15% weight) - Important for execution
            liquidity_score = max(0, 1.0 - (contract.bid_ask_spread_pct / 10.0))
            score += liquidity_score * 0.15
            
            # 4. IV level (15% weight) - Lower IV = better value
            if contract.implied_volatility:
                iv_score = max(0, 1.0 - contract.implied_volatility)
                score += iv_score * 0.15
            else:
                score += 0.0  # No IV data = neutral score
            
            # 5. Moneyness (10% weight) - ITM preference for LEAPs
            moneyness_score = self._calculate_moneyness_score(contract, market_data)
            score += moneyness_score * 0.10
            
            # 6. Risk-adjusted return potential (10% weight)
            return_score = self._calculate_return_potential(contract, market_data, risk_tolerance)
            score += return_score * 0.10
            
            scored.append((contract, score))
        
        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  🔍 Top 3 scored contracts:")
        for i, (contract, score) in enumerate(scored[:3]):
            print(f"    {i+1}. ${contract.strike} {contract.expiration.strftime('%b %Y')} - Score: {score:.3f}")
        
        return scored
    
    def _calculate_regime_aware_dte_score(self, dte: int, regime: str) -> float:
        """Calculate DTE score based on market regime"""
        
        if regime == "RISK_ON":
            # RISK_ON: Prefer 12-24 months for optimal LEAP strategy
            optimal_min, optimal_max = 365, 730
        elif regime == "TRANSITION":
            # TRANSITION: Prefer longer expirations to wait out uncertainty
            optimal_min, optimal_max = 540, 1095  # 18-36 months
        else:  # RISK_OFF (shouldn't reach here due to filter)
            optimal_min, optimal_max = 730, 1095  # 24-36 months
        
        # Score based on how close to optimal range
        if optimal_min <= dte <= optimal_max:
            # Perfect range - score based on center preference
            center = (optimal_min + optimal_max) / 2
            distance_from_center = abs(dte - center) / (optimal_max - optimal_min)
            return 1.0 - distance_from_center * 0.5
        elif dte < optimal_min:
            # Too short - penalize heavily
            penalty = (optimal_min - dte) / optimal_min
            return max(0, 0.3 - penalty * 0.3)
        else:
            # Too long - penalize moderately
            penalty = (dte - optimal_max) / optimal_max
            return max(0, 0.7 - penalty * 0.7)
    
    def _calculate_moneyness_score(self, contract: OptionContract, market_data: MarketData) -> float:
        """Calculate moneyness score - ITM preferred for LEAPs"""
        if not contract.last_price or not contract.strike:
            return 0.5  # Neutral if missing data
        
        moneyness = contract.strike / contract.last_price
        
        if contract.in_the_money:
            # ITM - score based on how deep ITM
            if moneyness < 0.8:  # Deep ITM
                return 1.0
            elif moneyness < 0.95:  # Moderate ITM
                return 0.8
            else:  # Slightly ITM
                return 0.6
        else:
            # OTM - penalize based on how far OTM
            if moneyness < 1.05:  # Slightly OTM
                return 0.4
            elif moneyness < 1.15:  # Moderately OTM
                return 0.2
            else:  # Far OTM
                return 0.0
    
    def _calculate_return_potential(
        self,
        contract: OptionContract,
        market_data: MarketData,
        risk_tolerance: str
    ) -> float:
        """Calculate potential return based on risk tolerance"""
        try:
            # Estimate 1-year return potential
            current_price = market_data.spot_price
            strike = contract.strike
            
            # Different return expectations based on risk tolerance
            if risk_tolerance == "CONSERVATIVE":
                expected_return = 0.15  # 15% annual
            elif risk_tolerance == "MODERATE":
                expected_return = 0.25  # 25% annual
            else:  # AGGRESSIVE
                expected_return = 0.40  # 40% annual
            
            # Calculate expected option value after 1 year
            future_price = current_price * (1 + expected_return)
            intrinsic_value = max(0, future_price - strike)
            
            # Current option price (use mid of bid/ask)
            current_option_price = (contract.bid + contract.ask) / 2
            
            if current_option_price > 0:
                return_potential = intrinsic_value / current_option_price - 1
                # Normalize to 0-1 score
                return max(0, min(1, return_potential / 2.0))
            
            return 0.0
            
        except Exception:
            return 0.5  # Default neutral score
    
    def _analyze_contract(self, contract: OptionContract, market_data: MarketData) -> Dict[str, any]:
        """Analyze the recommended contract"""
        try:
            # Calculate Greeks using Black-Scholes
            risk_free_rate = 0.05  # 5% risk-free rate (simplified)
            
            greeks = black_scholes_greeks(
                S=market_data.spot_price,
                K=contract.strike,
                T=contract.days_to_expiry / 365.0,
                r=risk_free_rate,
                sigma=contract.implied_volatility or 0.3,
                option_type='call'
            )
            
            # Calculate breakeven price
            premium = (contract.bid + contract.ask) / 2
            breakeven = contract.strike + premium
            
            # Calculate max loss (premium paid)
            max_loss = premium * 100  # Per contract
            
            # Calculate return scenarios
            scenarios = self._calculate_return_scenarios(contract, market_data)
            
            return {
                'greeks': greeks,
                'breakeven_price': breakeven,
                'max_loss_per_contract': max_loss,
                'premium_per_contract': premium * 100,
                'return_scenarios': scenarios,
                'time_decay_daily': greeks.get('theta', 0) * 100,
                'delta_exposure': greeks.get('delta', 0) * 100,
                'leverage_ratio': market_data.spot_price / premium if premium > 0 else 0
            }
            
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _calculate_return_scenarios(
        self,
        contract: OptionContract,
        market_data: MarketData
    ) -> Dict[str, Dict[str, float]]:
        """Calculate returns for different price scenarios"""
        try:
            current_price = market_data.spot_price
            strike = contract.strike
            premium = (contract.bid + contract.ask) / 2
            
            scenarios = {
                'bull_20pct': {'price_change': 0.20, 'description': '+20% move'},
                'bull_30pct': {'price_change': 0.30, 'description': '+30% move'},
                'base_10pct': {'price_change': 0.10, 'description': '+10% move'},
                'bear_10pct': {'price_change': -0.10, 'description': '-10% move'},
                'bear_20pct': {'price_change': -0.20, 'description': '-20% move'}
            }
            
            for scenario_name, scenario_data in scenarios.items():
                future_price = current_price * (1 + scenario_data['price_change'])
                option_value = max(0, future_price - strike)
                return_pct = (option_value - premium) / premium if premium > 0 else -1.0
                
                scenarios[scenario_name].update({
                    'future_stock_price': future_price,
                    'option_value': option_value,
                    'return_pct': return_pct * 100,
                    'profit_loss': (option_value - premium) * 100
                })
            
            return scenarios
            
        except Exception as e:
            return {'error': f'Scenario calculation failed: {str(e)}'}
    
    def _assess_risk(self, contract: OptionContract, market_data: MarketData) -> Dict[str, any]:
        """Assess risk level of the contract"""
        try:
            risk_factors = {}
            risk_score = 0.0
            
            # IV risk
            if contract.implied_volatility:
                if contract.implied_volatility > 0.5:
                    iv_risk = "HIGH"
                    risk_score += 0.3
                elif contract.implied_volatility > 0.3:
                    iv_risk = "MODERATE"
                    risk_score += 0.2
                else:
                    iv_risk = "LOW"
                    risk_score += 0.1
            else:
                iv_risk = "UNKNOWN"
            
            risk_factors['implied_volatility'] = iv_risk
            
            # Liquidity risk
            if contract.bid_ask_spread_pct > 5:
                liquidity_risk = "HIGH"
                risk_score += 0.3
            elif contract.bid_ask_spread_pct > 2:
                liquidity_risk = "MODERATE"
                risk_score += 0.2
            else:
                liquidity_risk = "LOW"
                risk_score += 0.1
            
            risk_factors['liquidity'] = liquidity_risk
            
            # Time risk
            if contract.days_to_expiry < 400:
                time_risk = "HIGH"
                risk_score += 0.2
            elif contract.days_to_expiry < 600:
                time_risk = "MODERATE"
                risk_score += 0.1
            else:
                time_risk = "LOW"
            
            risk_factors['time_decay'] = time_risk
            
            # Moneyness risk
            if contract.delta < 0.6:
                moneyness_risk = "HIGH"
                risk_score += 0.2
            elif contract.delta < 0.7:
                moneyness_risk = "MODERATE"
                risk_score += 0.1
            else:
                moneyness_risk = "LOW"
            
            risk_factors['moneyness'] = moneyness_risk
            
            # Overall risk tier
            if risk_score >= 0.8:
                risk_tier = "HIGH"
            elif risk_score >= 0.5:
                risk_tier = "MODERATE"
            else:
                risk_tier = "LOW"
            
            return {
                'risk_tier': risk_tier,
                'risk_score': round(risk_score, 2),
                'risk_factors': risk_factors,
                'max_loss_percent': 100.0,  # Options can lose 100%
                'probability_of_total_loss': self._estimate_total_loss_probability(contract)
            }
            
        except Exception as e:
            return {'error': f'Risk assessment failed: {str(e)}'}
    
    def _estimate_total_loss_probability(self, contract: OptionContract) -> float:
        """Estimate probability of total loss (option expires worthless)"""
        try:
            # Simplified calculation based on moneyness and time
            if contract.delta:
                # Delta approximates probability of finishing ITM
                itm_probability = contract.delta
                total_loss_probability = 1.0 - itm_probability
            else:
                # Use moneyness as proxy
                # This is very simplified - in production would use more sophisticated models
                total_loss_probability = 0.3  # Default 30% for LEAPs
            
            return round(total_loss_probability * 100, 1)
            
        except Exception:
            return 30.0  # Default estimate
    
    def _generate_suggestions(self, market_data: MarketData) -> List[str]:
        """Generate suggestions when no suitable contracts are found"""
        suggestions = []
        
        if not market_data.option_contracts:
            suggestions.append("No options available for this ticker")
            return suggestions
        
        # Analyze what's wrong with available contracts
        low_delta_count = sum(1 for c in market_data.option_contracts if c.delta and c.delta < 0.6)
        high_iv_count = sum(1 for c in market_data.option_contracts 
                          if c.implied_volatility and c.implied_volatility > 0.5)
        wide_spread_count = sum(1 for c in market_data.option_contracts 
                               if c.bid_ask_spread_pct > 5)
        
        if low_delta_count > len(market_data.option_contracts) * 0.7:
            suggestions.append("Consider lowering your target delta to 0.6")
        
        if high_iv_count > len(market_data.option_contracts) * 0.5:
            suggestions.append("IV is high - consider waiting for lower volatility")
        
        if wide_spread_count > len(market_data.option_contracts) * 0.5:
            suggestions.append("Low liquidity - consider larger cap stocks")
        
        # Check expiration timing
        short_dte_count = sum(1 for c in market_data.option_contracts if c.days_to_expiry < 400)
        if short_dte_count > len(market_data.option_contracts) * 0.8:
            suggestions.append("Consider longer expirations (18+ months)")
        
        return suggestions


# Convenience function for quick recommendations
def recommend_leap_structure(
    ticker: str,
    target_delta: float = 0.7,
    max_iv_percentile: float = 70.0,
    risk_tolerance: str = "MODERATE"
) -> Dict[str, any]:
    """
    Quick LEAP structure recommendation
    
    Args:
        ticker: Stock symbol
        target_delta: Target delta (0.7 default)
        max_iv_percentile: Maximum IV percentile
        risk_tolerance: CONSERVATIVE, MODERATE, or AGGRESSIVE
    
    Returns:
        Recommendation dictionary
    """
    from data.market_data_client import MarketDataClient
    
    client = MarketDataClient()
    selector = OptionSelector()
    
    # Fetch market data
    market_data = client.fetch_market_data(ticker)
    if not market_data:
        return {
            'success': False,
            'message': f'Could not fetch market data for {ticker}'
        }
    
    # Get recommendation
    return selector.recommend_optimal_structure(
        ticker=ticker,
        market_data=market_data,
        target_delta=target_delta,
        max_iv_percentile=max_iv_percentile,
        risk_tolerance=risk_tolerance
    )
