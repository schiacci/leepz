"""
Quantitative Calculations for LEAP Strategic Asset Engine

Implements Black-Scholes pricing, weighted scoring engine, and Monte Carlo simulations for option analysis.
"""
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, List
import re
import yfinance as yf
from datetime import datetime, timedelta


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float,
                       option_type: str = 'call') -> float:
    """
    Calculate Black-Scholes option price

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'

    Returns:
        Option price
    """
    if T <= 0:
        # At expiration
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

    return price


def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float,
                        option_type: str = 'call') -> Dict[str, float]:
    """
    Calculate Black-Scholes Greeks

    Returns:
        Dictionary with delta, gamma, theta, vega, rho
    """
    if T <= 0:
        # At expiration
        delta = 1.0 if (option_type == 'call' and S > K) else (0.0 if option_type == 'call' else -1.0 if S < K else 0.0)
        return {
            'delta': delta,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }

    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    if option_type == 'call':
        delta = stats.norm.cdf(d1)
    else:
        delta = stats.norm.cdf(d1) - 1

    # Gamma (same for calls and puts)
    gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta
    if option_type == 'call':
        theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * stats.norm.cdf(d2))
    else:
        theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * stats.norm.cdf(-d2))

    # Vega (same for calls and puts)
    vega = S * stats.norm.pdf(d1) * np.sqrt(T)

    # Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2)
    else:
        rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2)

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }


def extract_growth_assumptions(narrative: str) -> Dict[str, float]:
    """
    Extract growth assumptions from investment narrative

    Returns:
        Dictionary with growth rates and volatility assumptions
    """
    assumptions = {
        'expected_growth': 0.10,  # Default 10% annual growth
        'volatility': 0.30,       # Default 30% volatility
        'time_horizon': 1.0       # Default 1 year
    }

    # Look for percentage mentions in narrative
    percentages = re.findall(r'(\d+(?:\.\d+)?)%', narrative.lower())

    if percentages:
        # Assume first percentage is expected growth
        assumptions['expected_growth'] = float(percentages[0]) / 100

        # Look for volatility mentions
        vol_keywords = ['volatility', 'vol', 'sigma', 'risk']
        for i, pct in enumerate(percentages):
            # Check context around percentage
            start_idx = narrative.lower().find(pct + '%')
            if start_idx > 0:
                context = narrative[max(0, start_idx-50):start_idx+50].lower()
                if any(keyword in context for keyword in vol_keywords):
                    assumptions['volatility'] = float(pct) / 100
                    break

    # Look for specific growth mentions
    growth_patterns = [
        r'(\d+(?:\.\d+)?)%?\s*(?:annual\s*)?(?:growth|return|cagr)',
        r'(?:expect|project|forecast).*\s+(\d+(?:\.\d+)?)%',
        r'(?:grow|increase).*?\s+(\d+(?:\.\d+)?)%'
    ]

    for pattern in growth_patterns:
        matches = re.findall(pattern, narrative.lower())
        if matches:
            assumptions['expected_growth'] = float(matches[0].replace('%', '')) / 100
            break

    return assumptions


def monte_carlo_itm_probability(S: float, K: float, T: float, r: float, sigma: float,
                              growth_assumptions: Dict[str, float], n_runs: int = 1000) -> Dict[str, float]:
    """
    Run Monte Carlo simulation to calculate ITM probability

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Current volatility
        growth_assumptions: Dict with expected_growth, volatility, time_horizon
        n_runs: Number of simulation runs

    Returns:
        Dictionary with probability statistics
    """
    # Extract assumptions
    mu = growth_assumptions.get('expected_growth', 0.10)  # Expected growth rate
    vol = growth_assumptions.get('volatility', sigma)      # Use narrative vol or market vol
    dt = T / 252  # Daily time steps (trading days)

    # Simulate stock price paths using Geometric Brownian Motion
    np.random.seed(42)  # For reproducibility

    # Generate random returns
    Z = np.random.standard_normal((n_runs, int(252 * T)))
    daily_returns = (r - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z

    # Calculate cumulative returns
    cumulative_returns = np.exp(np.cumsum(daily_returns, axis=1))

    # Final stock prices
    final_prices = S * cumulative_returns[:, -1]

    # Check ITM (for calls: final_price > strike)
    itm_count = np.sum(final_prices > K)
    itm_probability = itm_count / n_runs

    # Calculate confidence intervals
    # Using normal approximation for binomial proportion
    se = np.sqrt(itm_probability * (1 - itm_probability) / n_runs)
    ci_lower = max(0, itm_probability - 1.96 * se)
    ci_upper = min(1, itm_probability + 1.96 * se)

    # Expected value of option at expiration
    itm_prices = final_prices[final_prices > K]
    if len(itm_prices) > 0:
        expected_itm_value = np.mean(itm_prices - K)
    else:
        expected_itm_value = 0.0

    return {
        'itm_probability': itm_probability,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'expected_itm_value': expected_itm_value,
        'n_runs': n_runs,
        'simulated_volatility': vol,
        'drift_rate': mu
    }


def verify_option_pricing(market_price: float, bs_price: float) -> Dict[str, any]:
    """
    Verify market price against Black-Scholes calculation

    Returns:
        Dictionary with pricing comparison and analysis
    """
    price_diff = market_price - bs_price
    price_diff_pct = price_diff / market_price if market_price != 0 else 0

    # Simple fair value assessment
    if abs(price_diff_pct) < 0.05:  # Within 5%
        fairness = "FAIR"
    elif price_diff_pct > 0.05:
        fairness = "OVERPRICED"
    else:
        fairness = "UNDERPRICED"

    return {
        'market_price': market_price,
        'bs_price': bs_price,
        'price_diff': price_diff,
        'price_diff_pct': price_diff_pct,
        'fairness_assessment': fairness
    }


def calculate_weighted_score(ticker: str, market_data: Dict, growth_data: Dict = None) -> Dict[str, any]:
    """
    Calculate weighted scoring model instead of binary buy/sell signals
    
    Args:
        ticker: Stock symbol
        market_data: Dictionary with market metrics
        growth_data: Dictionary with growth metrics (EPS, revenue, etc.)
    
    Returns:
        Dictionary with score (0-1), component breakdown, and recommendation
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Initialize component scores
        components = {}
        
        # 1. Trend Score (200MA + 50MA alignment) - Weight: 20%
        trend_score = calculate_trend_score(stock)
        components['trend'] = {
            'score': trend_score,
            'weight': 0.20,
            'description': '200MA + 50MA alignment'
        }
        
        # 2. Growth Score (EPS CAGR, revenue growth) - Weight: 25%
        growth_score = calculate_growth_score(info, growth_data)
        components['growth'] = {
            'score': growth_score,
            'weight': 0.25,
            'description': 'EPS CAGR, revenue growth'
        }
        
        # 3. Valuation Score (PEG, forward PE vs 5yr avg) - Weight: 20%
        valuation_score = calculate_valuation_score(info)
        components['valuation'] = {
            'score': valuation_score,
            'weight': 0.20,
            'description': 'PEG, forward PE vs 5yr avg'
        }
        
        # 4. Volatility Score (IV percentile) - Weight: 15%
        iv_percentile = market_data.get('iv_percentile', 50)
        volatility_score = max(0, 1 - (iv_percentile / 100))  # Lower IV = higher score
        components['volatility'] = {
            'score': volatility_score,
            'weight': 0.15,
            'description': 'IV percentile (lower is better)'
        }
        
        # 5. Momentum Score (6-12 month relative strength) - Weight: 10%
        momentum_score = calculate_momentum_score(stock)
        components['momentum'] = {
            'score': momentum_score,
            'weight': 0.10,
            'description': '6-12 month relative strength'
        }
        
        # 6. Market Regime Score - Weight: 10%
        regime_score = calculate_regime_score()
        components['regime'] = {
            'score': regime_score,
            'weight': 0.10,
            'description': 'Market regime filter'
        }
        
        # Calculate weighted total score
        total_score = sum(comp['score'] * comp['weight'] for comp in components.values())
        
        # Generate recommendation
        if total_score >= 0.8:
            recommendation = "STRONG LEAP BUY"
        elif total_score >= 0.7:
            recommendation = "LEAP BUY"
        elif total_score >= 0.6:
            recommendation = "WEAK LEAP BUY"
        elif total_score >= 0.4:
            recommendation = "HOLD"
        else:
            recommendation = "AVOID"
        
        return {
            'score': round(total_score, 3),
            'components': components,
            'recommendation': recommendation,
            'ticker': ticker
        }
        
    except Exception as e:
        print(f"Error calculating weighted score for {ticker}: {e}")
        return {
            'score': 0.0,
            'components': {},
            'recommendation': 'ERROR',
            'ticker': ticker
        }


def calculate_trend_score(stock: yf.Ticker) -> float:
    """Calculate trend score based on moving average alignment"""
    try:
        # Get 1 year of historical data
        hist = stock.history(period="1y")
        if hist.empty:
            return 0.5
        
        current_price = hist['Close'].iloc[-1]
        ma_50 = hist['Close'].rolling(50).mean().iloc[-1]
        ma_200 = hist['Close'].rolling(200).mean().iloc[-1]
        
        # Scoring logic
        score = 0.5  # Base score
        
        # Above 200MA is bullish
        if current_price > ma_200:
            score += 0.3
        
        # 50MA above 200MA is bullish
        if ma_50 > ma_200:
            score += 0.1
        
        # Current price above 50MA is bullish
        if current_price > ma_50:
            score += 0.1
        
        return min(1.0, score)
        
    except Exception:
        return 0.5


def calculate_growth_score(info: Dict, growth_data: Dict = None) -> float:
    """Calculate growth score based on EPS and revenue metrics"""
    try:
        score = 0.5  # Base score
        
        # EPS growth
        eps_growth = info.get('earningsGrowth', 0)
        if eps_growth and eps_growth > 0:
            score += min(0.25, eps_growth * 2)  # Cap at 0.25
        
        # Revenue growth
        revenue_growth = info.get('revenueGrowth', 0)
        if revenue_growth and revenue_growth > 0:
            score += min(0.25, revenue_growth * 2)  # Cap at 0.25
        
        return min(1.0, score)
        
    except Exception:
        return 0.5


def calculate_valuation_score(info: Dict) -> float:
    """Calculate valuation score based on PEG and PE ratios"""
    try:
        score = 0.5  # Base score
        
        # PEG ratio (lower is better, ideal < 1)
        peg = info.get('pegRatio')
        if peg and peg > 0:
            if peg <= 1:
                score += 0.3
            elif peg <= 1.5:
                score += 0.2
            elif peg <= 2:
                score += 0.1
        
        # Forward PE
        forward_pe = info.get('forwardPE')
        trailing_pe = info.get('trailingPE')
        
        if forward_pe and trailing_pe:
            # Forward PE lower than trailing PE is bullish
            if forward_pe < trailing_pe:
                score += 0.1
            
            # Reasonable PE range (10-25)
            if 10 <= forward_pe <= 25:
                score += 0.1
        
        return min(1.0, score)
        
    except Exception:
        return 0.5


def calculate_momentum_score(stock: yf.Ticker) -> float:
    """Calculate momentum score based on 6-12 month performance"""
    try:
        # Get 1 year of historical data
        hist = stock.history(period="1y")
        if hist.empty:
            return 0.5
        
        # Calculate 6-month and 12-month returns
        current_price = hist['Close'].iloc[-1]
        price_6m_ago = hist['Close'].iloc[-126] if len(hist) > 126 else hist['Close'].iloc[0]
        price_12m_ago = hist['Close'].iloc[0]
        
        return_6m = (current_price - price_6m_ago) / price_6m_ago
        return_12m = (current_price - price_12m_ago) / price_12m_ago
        
        score = 0.5  # Base score
        
        # Positive momentum
        if return_6m > 0:
            score += 0.15
        if return_12m > 0:
            score += 0.15
        
        # Strong momentum (>20% annual)
        if return_12m > 0.2:
            score += 0.2
        
        return min(1.0, score)
        
    except Exception:
        return 0.5


def calculate_regime_score() -> float:
    """Calculate market regime score"""
    try:
        # Get S&P 500 data
        spy = yf.Ticker("SPY")
        hist = spy.history(period="1y")
        
        if hist.empty:
            return 0.5
        
        current_price = hist['Close'].iloc[-1]
        ma_200 = hist['Close'].rolling(200).mean().iloc[-1]
        
        # Base score
        score = 0.5
        
        # Above 200MA is risk-on
        if current_price > ma_200:
            score += 0.3
        
        # Check VIX (optional enhancement)
        try:
            vix = yf.Ticker("^VIX")
            vix_current = vix.history(period="1mo")['Close'].iloc[-1]
            
            # Lower VIX is risk-on
            if vix_current < 20:
                score += 0.2
            elif vix_current < 30:
                score += 0.1
        except:
            pass
        
        return min(1.0, score)
        
    except Exception:
        return 0.5


def get_market_regime() -> str:
    """Determine current market regime using S&P 500, NASDAQ-100, and VIX"""
    try:
        # Get S&P 500 data
        spy = yf.Ticker("SPY")
        hist = spy.history(period="1y")
        
        if hist.empty:
            return "TRANSITION"
        
        current_price = hist['Close'].iloc[-1]
        ma_200 = hist['Close'].rolling(200).mean().iloc[-1]
        
        # Get NASDAQ-100 data
        qqq = yf.Ticker("QQQ")
        qqq_hist = qqq.history(period="1y")
        
        qqq_current = None
        qqq_ma_200 = None
        if not qqq_hist.empty:
            qqq_current = qqq_hist['Close'].iloc[-1]
            qqq_ma_200 = qqq_hist['Close'].rolling(200).mean().iloc[-1]
        
        # Get VIX
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="1y")
        
        vix_current = None
        vix_percentile = None
        if not vix_hist.empty:
            vix_current = vix_hist['Close'].iloc[-1]
            # Calculate VIX percentile (52-week)
            vix_values = vix_hist['Close'].values
            vix_percentile = (vix_values < vix_current).mean() * 100
        
        # Enhanced regime determination
        spy_above_ma = current_price > ma_200
        qqq_above_ma = qqq_current > qqq_ma_200 if qqq_current and qqq_ma_200 else True
        vix_low = vix_current < 20 if vix_current else True
        vix_high = vix_current > 30 if vix_current else False
        
        # Risk ON: Both indices above 200MA AND VIX < 20
        if spy_above_ma and qqq_above_ma and vix_low:
            return "RISK_ON"
        # Risk OFF: Either index below 200MA AND VIX > 30
        elif (not spy_above_ma or not qqq_above_ma) and vix_high:
            return "RISK_OFF"
        # TRANSITION: Mixed signals
        else:
            return "TRANSITION"
            
    except Exception as e:
        print(f"Error determining market regime: {e}")
        return "TRANSITION"


def get_market_regime_details() -> Dict[str, any]:
    """Get detailed market regime information"""
    try:
        details = {}
        
        # S&P 500 metrics
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="1y")
        if not spy_hist.empty:
            spy_current = spy_hist['Close'].iloc[-1]
            spy_ma_200 = spy_hist['Close'].rolling(200).mean().iloc[-1]
            spy_above_200ma = spy_current > spy_ma_200
            
            details['spy'] = {
                'current_price': spy_current,
                'ma_200': spy_ma_200,
                'above_200ma': spy_above_200ma,
                'distance_pct': ((spy_current - spy_ma_200) / spy_ma_200) * 100
            }
        
        # NASDAQ-100 metrics
        qqq = yf.Ticker("QQQ")
        qqq_hist = qqq.history(period="1y")
        if not qqq_hist.empty:
            qqq_current = qqq_hist['Close'].iloc[-1]
            qqq_ma_200 = qqq_hist['Close'].rolling(200).mean().iloc[-1]
            qqq_above_200ma = qqq_current > qqq_ma_200
            
            details['qqq'] = {
                'current_price': qqq_current,
                'ma_200': qqq_ma_200,
                'above_200ma': qqq_above_200ma,
                'distance_pct': ((qqq_current - qqq_ma_200) / qqq_ma_200) * 100
            }
        
        # VIX metrics
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="1y")
        if not vix_hist.empty:
            vix_current = vix_hist['Close'].iloc[-1]
            vix_values = vix_hist['Close'].values
            vix_percentile = (vix_values < vix_current).mean() * 100
            vix_ma_50 = vix_hist['Close'].rolling(50).mean().iloc[-1]
            
            details['vix'] = {
                'current': vix_current,
                'percentile': vix_percentile,
                'ma_50': vix_ma_50,
                'status': 'LOW' if vix_current < 20 else 'HIGH' if vix_current > 30 else 'NORMAL'
            }
        
        details['regime'] = get_market_regime()
        details['timestamp'] = datetime.now().isoformat()
        
        return details
        
    except Exception as e:
        print(f"Error getting market regime details: {e}")
        return {'regime': 'TRANSITION', 'error': str(e)}


def calculate_expected_value(
    ticker: str,
    option_contract: Dict,
    market_data: Dict,
    scenario_weights: Dict[str, float] = None
) -> Dict[str, any]:
    """
    Calculate Expected Value (EV) using 3-scenario model
    
    Args:
        ticker: Stock symbol
        option_contract: Option contract details
        market_data: Market data including current price
        scenario_weights: Custom probabilities for scenarios
    
    Returns:
        Dictionary with EV calculation and scenario analysis
    """
    try:
        # Default scenario probabilities
        default_weights = {
            'bull': 0.25,    # 25% chance of +30% move
            'base': 0.50,    # 50% chance of +15% move  
            'bear': 0.25     # 25% chance of -20% move
        }
        
        if scenario_weights:
            # Validate custom weights sum to 1
            total_weight = sum(scenario_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                print(f"Warning: Scenario weights sum to {total_weight}, using defaults")
                weights = default_weights
            else:
                weights = scenario_weights
        else:
            weights = default_weights
        
        # Extract contract details
        current_price = market_data.get('spot_price', 0)
        strike = option_contract.get('strike', 0)
        premium = (option_contract.get('bid', 0) + option_contract.get('ask', 0)) / 2
        days_to_expiry = option_contract.get('days_to_expiry', 365)
        implied_vol = option_contract.get('implied_volatility', 0.3)
        
        # Scenario definitions
        scenarios = {
            'bull': {
                'price_change': 0.30,    # +30%
                'description': 'Bull Case (+30%)',
                'probability': weights['bull']
            },
            'base': {
                'price_change': 0.15,    # +15%
                'description': 'Base Case (+15%)',
                'probability': weights['base']
            },
            'bear': {
                'price_change': -0.20,   # -20%
                'description': 'Bear Case (-20%)',
                'probability': weights['bear']
            }
        }
        
        # Calculate option payoffs for each scenario
        scenario_results = {}
        
        for scenario_name, scenario_data in scenarios.items():
            # Calculate future stock price
            future_price = current_price * (1 + scenario_data['price_change'])
            
            # Calculate option value at expiration using Black-Scholes
            # For simplicity, we'll use intrinsic value at expiration
            time_to_expiry = max(0, days_to_expiry / 365.0 - 1.0)  # 1 year forward
            
            if time_to_expiry > 0:
                # Use Black-Scholes for time value
                option_value = black_scholes_price(
                    S=future_price,
                    K=strike,
                    T=time_to_expiry,
                    r=0.05,  # Risk-free rate
                    sigma=implied_vol,
                    option_type='call'
                )
            else:
                # At expiration - intrinsic value only
                option_value = max(0, future_price - strike)
            
            # Calculate profit/loss
            profit_loss = option_value - premium
            return_pct = (profit_loss / premium) * 100 if premium > 0 else -100
            
            scenario_results[scenario_name] = {
                'future_stock_price': future_price,
                'option_value': option_value,
                'profit_loss': profit_loss,
                'return_pct': return_pct,
                'probability': scenario_data['probability'],
                'description': scenario_data['description'],
                'weighted_return': return_pct * scenario_data['probability']
            }
        
        # Calculate expected values
        expected_return_pct = sum(result['weighted_return'] for result in scenario_results.values())
        expected_option_value = sum(
            result['option_value'] * result['probability'] 
            for result in scenario_results.values()
        )
        expected_profit_loss = expected_option_value - premium
        
        # Risk metrics
        max_loss = -premium  # Maximum loss is premium paid
        max_gain = max(result['profit_loss'] for result in scenario_results.values())
        
        # Probability of positive return
        positive_return_prob = sum(
            result['probability'] for result in scenario_results.values()
            if result['profit_loss'] > 0
        )
        
        # Sharpe-like ratio (simplified)
        returns = [result['return_pct'] for result in scenario_results.values()]
        avg_return = np.mean(returns)
        return_std = np.std(returns)
        sharpe_ratio = avg_return / return_std if return_std > 0 else 0
        
        # Determine recommendation
        if expected_return_pct > 20 and positive_return_prob > 0.6:
            ev_recommendation = "STRONG BUY"
        elif expected_return_pct > 10 and positive_return_prob > 0.5:
            ev_recommendation = "BUY"
        elif expected_return_pct > 0:
            ev_recommendation = "WEAK BUY"
        else:
            ev_recommendation = "AVOID"
        
        return {
            'expected_value': {
                'expected_option_value': expected_option_value,
                'expected_profit_loss': expected_profit_loss,
                'expected_return_pct': expected_return_pct,
                'recommendation': ev_recommendation
            },
            'risk_metrics': {
                'max_loss': max_loss,
                'max_gain': max_gain,
                'positive_return_probability': positive_return_prob * 100,
                'sharpe_ratio': sharpe_ratio
            },
            'scenarios': scenario_results,
            'contract_details': {
                'ticker': ticker,
                'strike': strike,
                'premium': premium,
                'days_to_expiry': days_to_expiry,
                'current_stock_price': current_price
            },
            'edge_analysis': {
                'theoretical_edge': expected_return_pct,
                'margin_of_safety': (current_price - strike) / current_price if strike > 0 else 0,
                'time_decay_daily': -premium / days_to_expiry if days_to_expiry > 0 else 0
            }
        }
        
    except Exception as e:
        print(f"Error calculating expected value: {e}")
        return {
            'error': str(e),
            'expected_value': {'expected_return_pct': 0, 'recommendation': 'ERROR'}
        }


def calculate_probability_weighted_scenarios(
    ticker: str,
    current_price: float,
    volatility: float,
    time_horizon_years: float = 1.0
) -> Dict[str, any]:
    """
    Calculate probability-weighted scenarios using Monte Carlo simulation
    
    Args:
        ticker: Stock symbol
        current_price: Current stock price
        volatility: Annual volatility
        time_horizon_years: Time horizon in years
    
    Returns:
        Dictionary with probability distributions
    """
    try:
        # Run Monte Carlo simulation
        n_simulations = 10000
        dt = 1/252  # Daily time steps
        n_steps = int(time_horizon_years * 252)
        
        # Generate random paths
        np.random.seed(42)  # For reproducibility
        Z = np.random.standard_normal((n_simulations, n_steps))
        
        # Assume risk-neutral drift for simplicity
        drift = 0.05  # 5% risk-free rate
        daily_returns = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z
        
        # Calculate cumulative returns
        cumulative_returns = np.exp(np.cumsum(daily_returns, axis=1))
        final_prices = current_price * cumulative_returns[:, -1]
        
        # Calculate percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        price_percentiles = {
            f'p{p}': np.percentile(final_prices, p) for p in percentiles
        }
        
        # Calculate return percentiles
        return_percentiles = {
            f'p{p}': ((np.percentile(final_prices, p) - current_price) / current_price) * 100 
            for p in percentiles
        }
        
        # Define scenario ranges
        scenarios = {
            'extreme_bull': {
                'range': (90, 100),
                'probability': (final_prices > np.percentile(final_prices, 90)).mean() * 100,
                'description': 'Top 10% outcomes'
            },
            'bull': {
                'range': (75, 90),
                'probability': ((final_prices > np.percentile(final_prices, 75)) & 
                              (final_prices <= np.percentile(final_prices, 90))).mean() * 100,
                'description': '75-90th percentile'
            },
            'moderate_bull': {
                'range': (60, 75),
                'probability': ((final_prices > np.percentile(final_prices, 60)) & 
                              (final_prices <= np.percentile(final_prices, 75))).mean() * 100,
                'description': '60-75th percentile'
            },
            'neutral': {
                'range': (40, 60),
                'probability': ((final_prices > np.percentile(final_prices, 40)) & 
                              (final_prices <= np.percentile(final_prices, 60))).mean() * 100,
                'description': '40-60th percentile'
            },
            'moderate_bear': {
                'range': (25, 40),
                'probability': ((final_prices > np.percentile(final_prices, 25)) & 
                              (final_prices <= np.percentile(final_prices, 40))).mean() * 100,
                'description': '25-40th percentile'
            },
            'bear': {
                'range': (10, 25),
                'probability': ((final_prices > np.percentile(final_prices, 10)) & 
                              (final_prices <= np.percentile(final_prices, 25))).mean() * 100,
                'description': '10-25th percentile'
            },
            'extreme_bear': {
                'range': (0, 10),
                'probability': (final_prices <= np.percentile(final_prices, 10)).mean() * 100,
                'description': 'Bottom 10% outcomes'
            }
        }
        
        # Calculate probability of specific return thresholds
        return_thresholds = [-0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 1.0]
        return_probabilities = {
            f'return_gt_{int(r*100)}pct': (final_prices > current_price * (1 + r)).mean() * 100
            for r in return_thresholds
        }
        
        return {
            'simulation_parameters': {
                'n_simulations': n_simulations,
                'time_horizon_years': time_horizon_years,
                'volatility': volatility,
                'current_price': current_price
            },
            'price_percentiles': price_percentiles,
            'return_percentiles': return_percentiles,
            'scenarios': scenarios,
            'return_probabilities': return_probabilities,
            'statistics': {
                'mean_final_price': np.mean(final_prices),
                'median_final_price': np.median(final_prices),
                'std_final_price': np.std(final_prices),
                'probability_of_positive_return': (final_prices > current_price).mean() * 100,
                'probability_of_20pct_plus_return': (final_prices > current_price * 1.2).mean() * 100,
                'probability_of_50pct_plus_return': (final_prices > current_price * 1.5).mean() * 100
            }
        }
        
    except Exception as e:
        print(f"Error in Monte Carlo simulation: {e}")
        return {'error': str(e)}
