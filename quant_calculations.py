"""
Quantitative Calculations for LEAP Strategic Asset Engine

Implements Black-Scholes pricing and Monte Carlo simulations for option analysis.
"""
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import re


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
