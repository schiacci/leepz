"""
Market Data Client for LEAP Strategic Asset Engine
Uses yfinance to fetch live market data and option chains
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from models import MarketData, OptionContract
from config import config


class MarketDataClient:
    """
    Fetches real-time market data and option chains using yfinance
    Implements the "Eyes" component from the system design
    """
    
    def __init__(self):
        self.leap_heuristics = config.leap_heuristics
    
    def fetch_market_data(self, ticker: str) -> Optional[MarketData]:
        """
        Fetch comprehensive market data for a ticker
        
        Args:
            ticker: Stock symbol (e.g., "AAPL", "NVDA")
            
        Returns:
            MarketData object with spot price, options, and Greeks
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get current spot price
            info = stock.info
            spot_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if not spot_price:
                print(f"⚠️ Could not fetch spot price for {ticker}")
                return None
            
            # Fetch option chain
            option_contracts = self._fetch_leap_options(stock, spot_price)
            
            # Calculate IV rank (52-week percentile)
            iv_rank = self._calculate_iv_rank(stock)
            
            # Get 52-week high/low
            history = stock.history(period="1y")
            price_52w_high = history['High'].max() if not history.empty else None
            price_52w_low = history['Low'].min() if not history.empty else None
            
            return MarketData(
                ticker=ticker,
                spot_price=spot_price,
                implied_volatility_rank=iv_rank,
                option_contracts=option_contracts,
                price_52w_high=price_52w_high,
                price_52w_low=price_52w_low
            )
            
        except Exception as e:
            print(f"❌ Error fetching data for {ticker}: {e}")
            return None
    
    def _fetch_leap_options(self, stock: yf.Ticker, spot_price: float) -> List[OptionContract]:
        """
        Fetch LEAP options (expiry > 540 days) from the option chain
        
        Args:
            stock: yfinance Ticker object
            spot_price: Current stock price
            
        Returns:
            List of OptionContract objects filtered for LEAP criteria
        """
        try:
            expirations = stock.options
            if not expirations:
                print(f"  🔍 No option expirations found for {stock.ticker}")
                return []
            
            print(f"  🔍 Found {len(expirations)} expirations: {expirations[:5]}{'...' if len(expirations) > 5 else ''}")
            
            leap_contracts = []
            min_expiry_date = datetime.now() + timedelta(days=self.leap_heuristics.min_days_to_expiry)
            print(f"  🔍 Looking for expirations after: {min_expiry_date.strftime('%Y-%m-%d')}")
            
            for expiry_str in expirations:
                expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
                days_to_expiry = (expiry_date - datetime.now()).days
                
                # Only consider LEAP expirations (540+ days)
                if expiry_date < min_expiry_date:
                    continue
                
                print(f"  🔍 Processing expiry {expiry_str} ({days_to_expiry} days)...")
                
                # Fetch call options for this expiry
                try:
                    option_chain = stock.option_chain(expiry_str)
                    calls = option_chain.calls
                    
                    print(f"    📊 Found {len(calls)} call options for {expiry_str}")
                    
                    filtered_count = 0
                    spread_filtered = 0
                    delta_filtered = 0
                    no_greeks = 0
                    
                    for _, row in calls.iterrows():
                        # Calculate bid-ask spread
                        bid = row.get('bid', 0)
                        ask = row.get('ask', 0)
                        last = row.get('lastPrice', 0)
                        
                        if bid > 0 and ask > 0:
                            bid_ask_spread_pct = ((ask - bid) / ((ask + bid) / 2)) * 100
                        else:
                            continue  # Skip if no valid bid/ask
                        
                        # Check if within acceptable spread
                        if bid_ask_spread_pct > self.leap_heuristics.max_bid_ask_spread_pct:
                            spread_filtered += 1
                            continue
                        
                        # Extract Greeks (with fallback)
                        delta = row.get('delta')
                        
                        # If no delta from yfinance, estimate it using strike proximity
                        if not delta:
                            # Estimate delta based on how far strike is from spot price
                            # Deep ITM (strike << spot) ≈ delta 0.8-0.9
                            # ATM (strike ≈ spot) ≈ delta 0.5
                            # OTM (strike >> spot) ≈ delta 0.2-0.3
                            strike_to_spot_ratio = row['strike'] / spot_price
                            
                            if strike_to_spot_ratio <= 0.85:
                                delta = 0.85  # Deep ITM
                            elif strike_to_spot_ratio <= 0.95:
                                delta = 0.80  # ITM
                            elif strike_to_spot_ratio <= 1.05:
                                delta = 0.55  # ATM
                            else:
                                delta = 0.30  # OTM
                            
                            # Log that we're estimating
                            if filtered_count == 0 and len(leap_contracts) < 3:
                                print(f"    📐 Estimated delta {delta:.2f} for strike ${row['strike']} (ratio: {strike_to_spot_ratio:.2f})")
                        
                        # Only consider calls close to target delta (0.80)
                        # Allow range of 0.70 - 0.90 for flexibility
                        if 0.70 <= delta <= 0.90:
                            contract = OptionContract(
                                ticker=stock.ticker,
                                strike=row['strike'],
                                expiration=expiry_date,
                                days_to_expiry=days_to_expiry,
                                option_type='call',
                                last_price=last,
                                bid=bid,
                                ask=ask,
                                bid_ask_spread_pct=bid_ask_spread_pct,
                                delta=delta,
                                gamma=row.get('gamma'),
                                theta=row.get('theta'),
                                vega=row.get('vega'),
                                implied_volatility=row.get('impliedVolatility'),
                                volume=row.get('volume'),
                                open_interest=row.get('openInterest'),
                                in_the_money=row['strike'] < spot_price
                            )
                            leap_contracts.append(contract)
                            filtered_count += 1
                        else:
                            delta_filtered += 1
                    
                    print(f"    📊 Filter breakdown: {filtered_count} suitable, {spread_filtered} spread-filtered, {delta_filtered} delta-filtered, {no_greeks} no Greeks")
                    
                    # Show delta range if no suitable contracts found
                    if filtered_count == 0:
                        deltas = [row.get('delta') for _, row in calls.iterrows() if pd.notna(row.get('delta'))]
                        if deltas:
                            min_delta, max_delta = min(deltas), max(deltas)
                            print(f"    📊 Delta range: {min_delta:.3f} - {max_delta:.3f}")
                        
                        spreads = [((row.get('ask', 0) - row.get('bid', 0)) / ((row.get('ask', 0) + row.get('bid', 0)) / 2)) * 100 
                                 for _, row in calls.iterrows() if row.get('bid', 0) > 0 and row.get('ask', 0) > 0]
                        if spreads:
                            min_spread, max_spread = min(spreads), max(spreads)
                            print(f"    📊 Spread range: {min_spread:.1f}% - {max_spread:.1f}%")
                    
                    print(f"    ✅ Found {filtered_count} suitable contracts for {expiry_str}")
                            
                except Exception as e:
                    print(f"  ⚠️ Error processing expiry {expiry_str}: {e}")
                    continue
            
            print(f"  🔍 Total LEAP contracts found: {len(leap_contracts)}")
            
            # Sort by delta (closest to 0.80 first)
            leap_contracts.sort(key=lambda x: abs(x.delta - self.leap_heuristics.target_delta))
            
            return leap_contracts[:10]  # Return top 10 candidates
            
        except Exception as e:
            print(f"❌ Error fetching LEAP options: {e}")
            return []
    
    def _calculate_iv_rank(self, stock: yf.Ticker) -> Optional[float]:
        """
        Calculate implied volatility rank (IV percentile over 52 weeks)
        
        Returns:
            Float between 0-100 representing IV percentile
        """
        try:
            # Get historical options data (this is tricky with yfinance)
            # For now, we'll use a simplified approach
            # In production, you'd want to track historical IV data
            
            # Get current IV from ATM options
            expirations = stock.options
            if not expirations:
                return None
            
            # Use nearest expiration for current IV
            option_chain = stock.option_chain(expirations[0])
            calls = option_chain.calls
            
            if calls.empty:
                return None
            
            # Get ATM IV (strike closest to current price)
            current_price = stock.info.get('currentPrice', 0)
            atm_option = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
            current_iv = atm_option['impliedVolatility'].values[0] if not atm_option.empty else None
            
            # For simplicity, return 50 if we can't calculate historical range
            # In production, you'd maintain a database of historical IV
            return 50.0  # Placeholder
            
        except Exception as e:
            print(f"⚠️ Error calculating IV rank: {e}")
            return None
    
    def get_earnings_date(self, ticker: str) -> Optional[datetime]:
        """
        Check for upcoming earnings announcement
        
        Returns:
            Next earnings date if available
        """
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            if calendar is not None:
                # Handle both DataFrame and dict formats
                if hasattr(calendar, 'empty'):
                    # DataFrame format
                    if not calendar.empty:
                        earnings_date = calendar.get('Earnings Date')
                        if earnings_date is not None:
                            # Convert to datetime and ensure we get a single value
                            dt = pd.to_datetime(earnings_date)
                            if hasattr(dt, 'iloc'):
                                # It's a Series/Index, take the first value
                                dt = dt.iloc[0] if len(dt) > 0 else None
                            elif hasattr(dt, 'item'):
                                # It's a scalar array, get the scalar
                                dt = dt.item()
                            return dt if isinstance(dt, datetime) else None
                else:
                    # Dict format
                    earnings_date = calendar.get('Earnings Date')
                    if earnings_date is not None:
                        # Convert to datetime and ensure we get a single value
                        dt = pd.to_datetime(earnings_date)
                        if hasattr(dt, 'iloc'):
                            # It's a Series/Index, take the first value
                            dt = dt.iloc[0] if len(dt) > 0 else None
                        elif hasattr(dt, 'item'):
                            # It's a scalar array, get the scalar
                            dt = dt.item()
                        return dt if isinstance(dt, datetime) else None
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error fetching earnings date: {e}")
            return None
    
    def batch_fetch(self, tickers: List[str]) -> Dict[str, MarketData]:
        """
        Fetch market data for multiple tickers in batch
        
        Args:
            tickers: List of stock symbols
            
        Returns:
            Dictionary mapping ticker to MarketData
        """
        results = {}
        
        for ticker in tickers:
            print(f"📊 Fetching data for {ticker}...")
            data = self.fetch_market_data(ticker)
            if data:
                results[ticker] = data
        
        return results
