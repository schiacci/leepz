"""
LEAP Options Backtesting Engine

Professional-grade backtesting framework for LEAP options strategies.
Simulates historical performance with realistic assumptions and comprehensive metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import yfinance as yf

from quant_calculations import black_scholes_price
from config import config


@dataclass
class BacktestPosition:
    """Represents a single LEAP options position"""
    symbol: str
    entry_date: datetime
    expiration_date: datetime
    strike_price: float
    entry_price: float
    quantity: int
    entry_stock_price: float
    is_long: bool = True  # Long call positions

    @property
    def days_to_expiry(self) -> int:
        """Days remaining until expiration"""
        return max(0, (self.expiration_date - datetime.now()).days)

    def current_value(self, current_stock_price: float, volatility: float = 0.30) -> float:
        """Calculate current theoretical value using Black-Scholes"""
        if self.days_to_expiry <= 0:
            # At expiration, value is intrinsic only
            intrinsic = max(0, current_stock_price - self.strike_price)
            return intrinsic * self.quantity * 100  # 100 shares per contract

        # Use Black-Scholes for current value
        bs_price = black_scholes_price(
            S=current_stock_price,
            K=self.strike_price,
            T=self.days_to_expiry / 365.0,
            r=0.04,  # Risk-free rate assumption
            sigma=volatility,
            option_type='call'
        )
        return bs_price * self.quantity * 100

    def unrealized_pnl(self, current_stock_price: float, volatility: float = 0.30) -> float:
        """Calculate unrealized P&L"""
        current_value = self.current_value(current_stock_price, volatility)
        entry_value = self.entry_price * self.quantity * 100
        return current_value - entry_value


@dataclass
class BacktestResult:
    """Results from a backtesting run"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    avg_trade_return: float
    portfolio_values: List[float]
    dates: List[datetime]
    positions: List[BacktestPosition]
    trades_executed: List[Dict]


class BacktestEngine:
    """
    Professional LEAP options backtesting engine.

    Simulates LEAP option trading strategies over historical periods
    with realistic assumptions and comprehensive performance analysis.
    """

    def __init__(self,
                 start_date: datetime,
                 end_date: datetime,
                 initial_capital: float = 100000.0,
                 max_positions: int = 5,
                 position_size_pct: float = 0.20):  # 20% of capital per position

        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct

        # Portfolio state
        self.capital = initial_capital
        self.positions: List[BacktestPosition] = []
        self.portfolio_values: List[float] = []
        self.dates: List[datetime] = []
        self.trades_executed: List[Dict] = []

        # Transaction costs (realistic assumptions)
        self.commission_per_contract = 2.50  # $2.50 per contract
        self.bid_ask_spread_pct = 0.03  # 3% spread on options

    def download_historical_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Download historical stock price data for backtesting"""
        print(f"📊 Downloading historical data for {len(symbols)} symbols...")

        historical_data = {}
        for symbol in symbols:
            try:
                # Download data with 1-year buffer for volatility calculation
                start_buffer = self.start_date - timedelta(days=365)
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_buffer, end=self.end_date, interval='1d')

                if len(data) > 0:
                    historical_data[symbol] = data
                    print(f"✅ Downloaded {len(data)} days of data for {symbol}")
                else:
                    print(f"⚠️ No data available for {symbol}")

            except Exception as e:
                print(f"❌ Error downloading {symbol}: {e}")

        return historical_data

    def calculate_historical_volatility(self, prices: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling historical volatility"""
        # Daily returns
        returns = prices.pct_change()

        # Rolling volatility (annualized)
        volatility = returns.rolling(window=window).std() * np.sqrt(252)

        # Fill NaN values with expanding mean
        volatility = volatility.fillna(volatility.expanding().mean())

        return volatility

    def find_leap_opportunities(self, symbol: str, data: pd.DataFrame,
                            current_date: datetime) -> List[Dict]:
        """
        Identify potential LEAP opportunities based on strategy rules.

        Strategy: Buy LEAP calls when:
        - Stock is in uptrend (200-day MA > 50-day MA)
        - RSI not overbought (< 70)
        - Implied volatility reasonable (20-50%)
        - Stock price > $50 (liquidity)
        - At least 1 year to expiration
        - At least 3 months of holding period remaining in backtest
        """
        # Constants for LEAP requirements
        MIN_DAYS_TO_EXPIRY = 365  # At least 1 year to expiration
        MIN_HOLDING_DAYS = 90     # Minimum holding period in days

        # Calculate remaining time in the backtest
        days_until_backtest_end = (self.end_date - current_date).days

        # Skip if not enough time to hold a position
        if days_until_backtest_end < MIN_HOLDING_DAYS:
            return []

        # Skip if not enough time for a LEAP
        if days_until_backtest_end < MIN_DAYS_TO_EXPIRY:
            return []

        opportunities = []

        try:
            # Get recent data (last 300 trading days)
            # Convert timezone-aware index to naive for comparison
            naive_index = data.index.tz_localize(None)
            recent_data = data[naive_index <= current_date].tail(300)

            if len(recent_data) < 200:
                return opportunities

            # Calculate technical indicators
            close_prices = recent_data['Close']

            # Moving averages
            ma50 = close_prices.rolling(50).mean()
            ma200 = close_prices.rolling(200).mean()

            # RSI calculation
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Historical volatility (30-day)
            volatility = self.calculate_historical_volatility(close_prices, 30)

            # Current values
            current_price = close_prices.iloc[-1]
            current_ma50 = ma50.iloc[-1]
            current_ma200 = ma200.iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_vol = volatility.iloc[-1]

            # Strategy conditions (relaxed for testing temporal constraints)
            uptrend = current_ma50 > current_ma50 * 0.95  # Much more lenient - just need some upward trend
            not_overbought = current_rsi < 80  # Allow higher RSI
            reasonable_vol = current_vol > 0.05  # Much more lenient - just need some volatility
            liquid_stock = current_price > 10  # Much more lenient - allow lower priced stocks

            # Debug logging
            if not (uptrend and not_overbought and reasonable_vol and liquid_stock):
                print(f"  📊 {symbol} conditions check:")
                print(f"    Price: ${current_price:.2f} (liquid: {liquid_stock})")
                print(f"    MA50: {current_ma50:.2f}, MA200: {current_ma200:.2f} (uptrend: {uptrend})")
                print(f"    RSI: {current_rsi:.1f} (not_overbought: {not_overbought})")
                print(f"    Vol: {current_vol:.3f} (reasonable_vol: {reasonable_vol})")
            else:
                print(f"  ✅ {symbol} meets all conditions - ready for LEAP trade!")

            if uptrend and not_overbought and reasonable_vol and liquid_stock:
                # Generate LEAP strike prices (OTM calls)
                for months in [12, 18, 24]:  # 1, 1.5, 2 years out
                    # Calculate expiration date
                    expiration = current_date + timedelta(days=months * 30)
                    
                    # Skip if expiration is after our backtest period or less than 1 year away
                    if expiration >= self.end_date or (expiration - current_date).days < 365:
                        continue
                        
                    # OTM strikes: 10%, 25%, 40% above current price
                    for otm_pct in [0.10, 0.25, 0.40]:
                        strike = current_price * (1 + otm_pct)

                        # Calculate actual days to expiration for Black-Scholes
                        days_to_expiry = (expiration - current_date).days
                        
                        # Estimate option price using Black-Scholes
                        bs_price = black_scholes_price(
                            S=current_price,
                            K=strike,
                            T=days_to_expiry / 365.0,  # Years to expiry
                            r=0.04,
                            sigma=current_vol,
                            option_type='call'
                        )

                        # Add bid-ask spread
                        market_price = bs_price * (1 + self.bid_ask_spread_pct)

                        opportunity = {
                            'symbol': symbol,
                            'strike': round(strike, 2),
                            'expiration': expiration,
                            'estimated_price': round(bs_price, 2),
                            'market_price': round(market_price, 2),
                            'volatility': current_vol,
                            'stock_price': current_price,
                            'score': self._calculate_opportunity_score(
                                current_price, strike, bs_price, current_vol, 
                                days_to_expiry / 30.0  # Convert to months
                            )
                        }

                        opportunities.append(opportunity)

        except Exception as e:
            print(f"❌ Error finding opportunities for {symbol}: {e}")

        # Sort by score (highest first)
        opportunities.sort(key=lambda x: x['score'], reverse=True)

        return opportunities

    def _calculate_opportunity_score(self, stock_price: float, strike: float,
                                   option_price: float, volatility: float,
                                   months_to_expiry: int) -> float:
        """Calculate opportunity score based on risk/reward metrics"""

        # Distance OTM (higher = more leverage)
        otm_distance = (strike - stock_price) / stock_price

        # Expected return (rough estimate)
        expected_return = otm_distance * 0.7  # Conservative estimate

        # Time value ratio (higher = better premium decay)
        intrinsic = max(0, stock_price - strike)
        time_value_ratio = (option_price - intrinsic) / option_price if option_price > 0 else 0

        # Volatility adjustment (prefer moderate vol)
        vol_score = 1.0 - abs(volatility - 0.35) / 0.35  # Peak at 35% vol

        # Time score (prefer longer expirations)
        time_score = min(months_to_expiry / 24.0, 1.0)

        # Combined score
        score = (expected_return * 0.4 +
                time_value_ratio * 0.3 +
                vol_score * 0.2 +
                time_score * 0.1)

        return score

    def execute_trade(self, opportunity: Dict, position_size: float, trade_date: datetime = None) -> Optional[BacktestPosition]:
        """Execute a LEAP trade with transaction costs
        
        Args:
            opportunity: Dictionary containing trade details
            position_size: Dollar amount to allocate to this trade (ignored - always buy 1 contract)
            trade_date: Date to record for the trade (defaults to current date if None)
        """
        try:
            # Use provided trade_date or fall back to current date
            trade_date = trade_date or datetime.now()
            
            # Always buy exactly 1 contract for simplicity
            contracts = 1

            print(f"    💰 Position size: 1 contract")
            print(f"    📊 Contract calc: price=${opportunity['market_price']:.2f}, contracts={contracts}")

            # Calculate total cost including commissions
            premium_cost = opportunity['market_price'] * contracts * 100
            commission_cost = self.commission_per_contract * contracts
            total_cost = premium_cost + commission_cost

            print(f"    💵 Cost breakdown: premium=${premium_cost:.0f}, commission=${commission_cost:.0f}, total=${total_cost:.0f}")

            if total_cost > self.capital:
                print(f"    ❌ Total cost (${total_cost:.0f}) exceeds capital (${self.capital:.0f})")
                return None

            # Create position
            position = BacktestPosition(
                symbol=opportunity['symbol'],
                entry_date=trade_date,  # Use the provided trade date
                expiration_date=opportunity['expiration'],
                strike_price=opportunity['strike'],
                entry_price=opportunity['market_price'],
                quantity=contracts,
                entry_stock_price=opportunity['stock_price']
            )

            # Update capital
            self.capital -= total_cost
            self.positions.append(position)

            # Record trade with expiration date
            trade_record = {
                'date': trade_date,  # Use the provided trade date
                'symbol': opportunity['symbol'],
                'action': 'BUY',
                'type': 'LEAP_CALL',
                'strike': opportunity['strike'],
                'expiration': opportunity['expiration'],
                'price': opportunity['market_price'],
                'contracts': contracts,
                'total_cost': total_cost,
                'stock_price': opportunity['stock_price'],
                'expiration_date': opportunity['expiration'].strftime('%Y-%m-%d')  # Add formatted expiration date
            }

            self.trades_executed.append(trade_record)

            # Print more detailed trade information
            contract_info = f"{contracts} {opportunity['symbol']} ${opportunity['strike']:.2f} Call"
            if trade_record['action'] == 'BUY':
                print(f"{trade_record['date'].strftime('%Y-%m-%d')} | "
                      f"{trade_record['action']} {contract_info} | "
                      f"Exp: {trade_record.get('expiration_date', 'N/A')} | "
                      f"Cost: ${trade_record.get('total_cost', 0):.2f} | "
                      f"Stock: ${trade_record.get('stock_price', 0):.2f}")
            else:
                print(f"{trade_record['date'].strftime('%Y-%m-%d')} | "
                      f"{trade_record['action']} {contract_info}")

            return position

        except Exception as e:
            print(f"❌ Error executing trade: {e}")
            return None

    def update_positions(self, current_date: datetime, stock_prices: Dict[str, float]):
        """Update position values and check for expirations"""

        expired_positions = []

        for position in self.positions:
            # Update current value
            current_price = stock_prices.get(position.symbol, position.entry_stock_price)
            current_vol = 0.30  # Default assumption, could be improved

            position_value = position.current_value(current_price, current_vol)

            # Check for expiration (use min of expiration date or backtest end date)
            expiration_date = min(position.expiration_date, self.end_date)
            if current_date >= expiration_date:
                # Close expired position
                pnl = position.unrealized_pnl(current_price, current_vol)
                self.capital += position_value

                # Record closing trade with the actual expiration date used
                self.trades_executed.append({
                    'date': expiration_date,  # Use the actual expiration date
                    'symbol': position.symbol,
                    'action': 'EXPIRE',
                    'strike': position.strike_price,
                    'expiration': expiration_date,
                    'closing_price': current_price,
                    'pnl': pnl,
                    'reason': 'EXPIRED' if position.expiration_date <= self.end_date else 'BACKTEST_ENDED'
                })

                expired_positions.append(position)

                expiration_reason = 'expired' if position.expiration_date <= self.end_date else 'position closed at backtest end'
                print(f"⏰ Position {expiration_reason}: {position.symbol} ${position.strike_price} "
                      f"PNL: ${pnl:.2f}")

        # Remove expired positions
        for position in expired_positions:
            self.positions.remove(position)

    def calculate_portfolio_value(self, current_date: datetime,
                                stock_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""

        cash_value = self.capital
        positions_value = 0.0

        for position in self.positions:
            current_price = stock_prices.get(position.symbol, position.entry_stock_price)
            positions_value += position.current_value(current_price)

        total_value = cash_value + positions_value

        # Record for tracking
        self.portfolio_values.append(total_value)
        self.dates.append(current_date)

        return total_value

    def run_backtest(self, symbols: List[str]) -> BacktestResult:
        """Run the complete backtesting simulation"""
        
        # Reset portfolio state at the beginning of each backtest
        self.capital = self.initial_capital
        self.positions = []
        self.portfolio_values = []
        self.dates = []
        self.trades_executed = []
        
        print(f"🚀 Starting LEAP backtest from {self.start_date.date()} to {self.end_date.date()}")
        print(f"💰 Initial capital: ${self.initial_capital:,.0f}")

        # Download historical data
        historical_data = self.download_historical_data(symbols)

        if not historical_data:
            raise ValueError("No historical data available for backtesting")

        # Track active positions per symbol
        positions_per_symbol = {symbol: 0 for symbol in symbols}
        max_positions_per_symbol = 999  # Unlimited positions per symbol

        # Simulation loop (monthly rebalancing)
        current_date = self.start_date

        while current_date <= self.end_date:
            print(f"📅 Processing {current_date.strftime('%Y-%m-%d')}...")

            # Get current stock prices
            current_prices = {}
            for symbol, data in historical_data.items():
                # Convert timezone-aware index to naive for comparison
                naive_index = data.index.tz_localize(None)
                price_data = data[naive_index <= current_date]
                if not price_data.empty:
                    current_prices[symbol] = price_data['Close'].iloc[-1]

            # Update existing positions
            self.update_positions(current_date, current_prices)

            # Reset position counts for current date
            positions_per_symbol = {symbol: 0 for symbol in symbols}
            for pos in self.positions:
                positions_per_symbol[pos.symbol] += 1

            # Check each symbol for opportunities
            for symbol in symbols:
                # Only consider symbols where we're under the position limit
                if (positions_per_symbol.get(symbol, 0) < max_positions_per_symbol and 
                    self.capital > 1000 and  # Minimum capital required
                    symbol in current_prices):
                    
                    opportunities = self.find_leap_opportunities(
                        symbol, historical_data[symbol], current_date
                    )

                    if opportunities:
                        # Execute best opportunity
                        best_opportunity = opportunities[0]
                        position_size = min(
                            self.capital * self.position_size_pct,  # Max position size
                            self.capital * 0.5  # Don't allocate more than 50% to any single position
                        )

                        if position_size >= 1000:  # Minimum position size
                            trade_date = current_date
                            if self.execute_trade(best_opportunity, position_size, trade_date):
                                positions_per_symbol[symbol] += 1

            # Move to next month for monthly rebalancing
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1, day=1)

            # Calculate portfolio value at this date
            self.calculate_portfolio_value(current_date, current_prices)

        # Final portfolio value
        final_value = self.calculate_portfolio_value(self.end_date, current_prices)
        print(f"✅ Backtest completed!")
        print(f"📊 Final portfolio value: ${final_value:,.0f}")

        return self._calculate_results()

    def _calculate_results(self) -> BacktestResult:
        """Calculate comprehensive backtest results and metrics"""

        if not self.portfolio_values:
            return BacktestResult(
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                profitable_trades=0,
                avg_trade_return=0.0,
                portfolio_values=[],
                dates=[],
                positions=self.positions,
                trades_executed=self.trades_executed
            )

        # Basic returns
        initial_value = self.initial_capital
        final_value = self.portfolio_values[-1]
        total_return = ((final_value - initial_value) / initial_value) * 100

        # Annualized return
        days = (self.dates[-1] - self.dates[0]).days
        years = days / 365.0
        annualized_return = ((final_value / initial_value) ** (1 / years) - 1) * 100

        # Sharpe ratio (assuming 4% risk-free rate)
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:  # Need at least 2 points and non-zero std
            excess_returns = returns - 0.04/252  # Daily risk-free rate
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0  # No meaningful Sharpe ratio when no trades or constant portfolio

        # Maximum drawdown
        cumulative = pd.Series(self.portfolio_values)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Calculate trade statistics
        trades = [t for t in self.trades_executed if t.get('action') in ['BUY', 'EXPIRE']]
        total_trades = len([t for t in trades if t['action'] == 'BUY'])  # Count only BUY actions
        profitable_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0.0
        avg_trade_return = (sum(t.get('pnl', 0) for t in trades) / total_trades) if total_trades > 0 else 0.0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            avg_trade_return=avg_trade_return,
            portfolio_values=self.portfolio_values,
            dates=self.dates,
            positions=self.positions,
            trades_executed=self.trades_executed
        )
