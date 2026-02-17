"""
LEAP Strategic Asset Engine - Enhanced CLI with Advanced Parameters

Usage:
    # Basic analysis
    python main.py --symbols META,AAPL,MSFT --min-score 0.7
    
    # Advanced filtering
    python main.py --symbols NVDA --target-delta 0.8 --max-iv-percentile 40 --regime-filter on
    
    # Portfolio analysis
    python main.py --symbols META,AAPL,MSFT --portfolio-size 250000 --min-growth 15
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator
from config import config
from quant_calculations import calculate_weighted_score, get_market_regime
from option_selector import recommend_leap_structure
from data.market_data_client import MarketDataClient
from datetime import datetime


def setup_logging():
    """Setup logging configuration"""
    log_dir = config.logging.file.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"📋 Logging to: {config.logging.file}")


def print_enhanced_results(symbol: str, analysis_result: dict, args):
    """Print enhanced analysis results with new format"""
    print(f"\n{'='*80}")
    print(f"📊 LEAP ANALYSIS: {symbol}")
    print(f"{'='*80}")
    
    # Core metrics
    score = analysis_result.get('score', 0)
    recommendation = analysis_result.get('recommendation', 'UNKNOWN')
    regime = get_market_regime()
    
    print(f"Score: {score:.3f}")
    print(f"Regime: {regime}")
    print(f"Recommendation: {recommendation}")
    
    # Option structure if available
    if 'option_recommendation' in analysis_result:
        opt = analysis_result['option_recommendation']
        if opt.get('success'):
            contract = opt['recommendation']['contract']
            print(f"Strike: ${contract.strike} {contract.expiration.strftime('%b %Y')}")
            print(f"Delta: {contract.delta:.2f}")
            print(f"Risk Tier: {opt['recommendation']['risk_assessment']['risk_tier']}")
            
            # Position sizing
            if args.portfolio_size:
                allocation_pct = min(0.10, (score - 0.5) * 0.2)  # Scale allocation with score
                position_size = args.portfolio_size * allocation_pct
                print(f"Suggested Allocation: {allocation_pct*100:.1f}% (${position_size:,.0f})")
    
    print(f"{'='*80}")


def main():
    """Enhanced main entry point with new parameters"""
    parser = argparse.ArgumentParser(
        description="LEAP Strategic Asset Engine - Enhanced with weighted scoring and advanced filtering"
    )
    
    # Core analysis parameters
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of tickers to analyze (e.g., 'META,AAPL,MSFT')"
    )
    
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="Minimum weighted score to recommend (0.0-1.0, default: 0.6)"
    )
    
    parser.add_argument(
        "--max-iv-percentile",
        type=float,
        default=70.0,
        help="Maximum IV percentile allowed (default: 70.0)"
    )
    
    # Option structure parameters
    parser.add_argument(
        "--target-delta",
        type=float,
        default=0.7,
        help="Target option delta (0.5-0.9, default: 0.7)"
    )
    
    parser.add_argument(
        "--risk-tolerance",
        type=str,
        choices=["CONSERVATIVE", "MODERATE", "AGGRESSIVE"],
        default="MODERATE",
        help="Risk tolerance level (default: MODERATE)"
    )
    
    # Market filtering
    parser.add_argument(
        "--regime-filter",
        type=str,
        choices=["on", "off"],
        default="on",
        help="Enable market regime filter (default: on)"
    )
    
    parser.add_argument(
        "--min-growth",
        type=float,
        default=10.0,
        help="Minimum expected growth rate (default: 10.0)"
    )
    
    # Portfolio parameters
    parser.add_argument(
        "--portfolio-size",
        type=float,
        help="Total portfolio size for position sizing"
    )
    
    parser.add_argument(
        "--max-position-size",
        type=float,
        default=10.0,
        help="Maximum position size as percentage of portfolio (default: 10.0)"
    )
    
    # Existing parameters
    parser.add_argument(
        "--narrative",
        type=str,
        help="Investment thesis/narrative for manual ticker analysis"
    )
    
    parser.add_argument(
        "--ticker",
        type=str,
        help="Single ticker symbol to analyze (e.g., 'META')"
    )
    
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export approved trades to markdown files"
    )
    
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run LEAP options backtesting simulation"
    )
    
    parser.add_argument(
        "--backtest-symbols",
        type=str,
        default="NVDA,AAPL,MSFT,TSLA,GOOGL",
        help="Symbols to include in backtest (default: NVDA,AAPL,MSFT,TSLA,GOOGL)"
    )
    
    parser.add_argument(
        "--backtest-start",
        type=str,
        default="2020-01-01",
        help="Backtest start date (YYYY-MM-DD, default: 2020-01-01)"
    )
    
    parser.add_argument(
        "--backtest-end",
        type=str,
        default="2024-01-01",
        help="Backtest end date (YYYY-MM-DD, default: 2024-01-01)"
    )
    
    parser.add_argument(
        "--backtest-capital",
        type=float,
        default=100000,
        help="Initial capital for backtest (default: 100000)"
    )
    
    # New advanced options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed analysis breakdown"
    )
    
    parser.add_argument(
        "--stress-test",
        action="store_true",
        help="Run stress scenarios on selected symbols"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    
    # Check regime filter
    if args.regime_filter == "on":
        regime = get_market_regime()
        if regime == "RISK_OFF":
            print(f"⚠️ Market Regime: {regime} - Blocking new LEAP buys")
            if not input("Continue anyway? (y/n): ").lower().startswith('y'):
                return
        print(f"✅ Market Regime: {regime} - Proceeding with analysis")
    
    # Initialize components
    client = MarketDataClient()
    
    try:
        # Enhanced symbol analysis mode
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
            print(f"\n� Enhanced Analysis: {len(symbols)} symbols")
            print(f"Filters: Score≥{args.min_score}, IV≤{args.max_iv_percentile}%, Delta≈{args.target_delta}")
            
            approved_symbols = []
            
            for symbol in symbols:
                print(f"\n{'-'*60}")
                print(f"🔍 Analyzing {symbol}...")
                
                # Fetch market data
                market_data = client.fetch_market_data(symbol)
                if not market_data:
                    print(f"❌ No data available for {symbol}")
                    continue
                
                # Calculate weighted score
                score_result = calculate_weighted_score(symbol, {
                    'iv_percentile': market_data.implied_volatility_rank or 50
                })
                
                score = score_result['score']
                recommendation = score_result['recommendation']
                
                # Apply filters
                if score < args.min_score:
                    print(f"❌ Score too low: {score:.3f} < {args.min_score}")
                    continue
                
                if market_data.implied_volatility_rank and market_data.implied_volatility_rank > args.max_iv_percentile:
                    print(f"❌ IV too high: {market_data.implied_volatility_rank:.1f}% > {args.max_iv_percentile}%")
                    continue
                
                # Get option recommendation using already fetched market data
                from option_selector import OptionSelector
                selector = OptionSelector()
                option_rec = selector.recommend_optimal_structure(
                    ticker=symbol,
                    market_data=market_data,  # Use the market data we already fetched!
                    target_delta=args.target_delta,
                    min_dte=180,  # More flexible - 6 months minimum
                    max_dte=1095,  # More flexible - 3 years maximum
                    max_iv_percentile=args.max_iv_percentile,
                    risk_tolerance=args.risk_tolerance
                )
                
                # Debug option recommendation
                if not option_rec.get('success'):
                    print(f"  🔍 Option Recommendation Failed: {option_rec.get('message', 'Unknown error')}")
                    if 'debug_info' in option_rec:
                        debug = option_rec['debug_info']
                        print(f"  📊 Debug: {debug['total_contracts']} contracts, delta target {debug['target_delta']}")
                        print(f"  📊 Debug: DTE range {debug['min_dte']}-{debug['max_dte']}, max IV {debug['max_iv_percentile']}%")
                
                # Combine results
                analysis_result = {
                    'score': score,
                    'recommendation': recommendation,
                    'option_recommendation': option_rec,
                    'market_data': market_data,
                    'score_breakdown': score_result.get('components', {}) if args.verbose else None
                }
                
                # Print results
                print_enhanced_results(symbol, analysis_result, args)
                
                # Count as approved if score passes minimum, regardless of option recommendation success
                if score >= args.min_score:
                    approved_symbols.append((symbol, analysis_result))
            
            # Summary
            print(f"\n{'='*80}")
            print(f"📊 ANALYSIS COMPLETE: {len(approved_symbols)}/{len(symbols)} approved")
            print(f"{'='*80}")
            
            # Create summary table
            if approved_symbols:
                print(f"\n📋 LEAP OPPORTUNITIES SUMMARY")
                print(f"{'='*120}")
                print(f"{'Symbol':<8} | {'Score':<6} | {'Recommendation':<15} | {'LEAP Strike':<12} | {'Expiration':<12} | {'Delta':<6} | {'Risk Tier':<10}")
                print(f"{'-'*120}")
                
                for symbol, result in approved_symbols:
                    score = result['score']
                    rec = result['recommendation']
                    
                    # Get option details if available
                    strike = "N/A"
                    expiration = "N/A"
                    delta = "N/A"
                    risk_tier = "N/A"
                    
                    if result['option_recommendation'].get('success'):
                        opt = result['option_recommendation']['recommendation']
                        contract = opt['contract']
                        strike = f"${contract.strike}"
                        expiration = contract.expiration.strftime('%b %Y')
                        delta = f"{contract.delta:.2f}"
                        risk_tier = opt['risk_assessment']['risk_tier']
                    
                    print(f"{symbol:<8} | {score:<6.3f} | {rec:<15} | {strike:<12} | {expiration:<12} | {delta:<6} | {risk_tier:<10}")
                
                print(f"{'-'*120}")
                print(f"\n💡 Key:")
                print(f"• Score: 0-1 weighted rating (higher = better)")
                print(f"• Delta: Option sensitivity to stock price (0.7 = deep ITM LEAP)")
                print(f"• Risk Tier: LOW/MODERATE/HIGH based on historical drawdowns")
                print(f"• Only symbols meeting minimum score threshold shown")
            
            if approved_symbols:
                for symbol, result in approved_symbols:
                    print(f"\n{symbol}: {result['recommendation']}")
                    print(f"  Score: {result['score']:.3f}")
                    
                    # Only show option details if option recommendation was successful
                    if result['option_recommendation'].get('success'):
                        opt = result['option_recommendation']['recommendation']
                        contract = opt['contract']
                        print(f"  Contract: ${contract.strike} Call {contract.expiration.strftime('%b %Y')}")
                        print(f"  Delta: {contract.delta:.2f} | Risk: {opt['risk_assessment']['risk_tier']}")
                        print(f"  Score: {opt.get('score', 'N/A')}")
                        
                        # Calculate allocation only if option recommendation succeeded
                        allocation = min(args.max_position_size, 
                                      (result['score'] - 0.5) * 20) if args.portfolio_size else 0
                    else:
                        print(f"  Option Recommendation: Failed - check market data availability")
                        allocation = 0
                    
                    if allocation > 0:
                        position_value = args.portfolio_size * allocation / 100
                        print(f"  Allocation: {allocation:.1f}% (${position_value:,.0f})")
                
                # Export option
                if args.export:
                    export_path = Path("./outputs/enhanced_analysis.md")
                    export_path.parent.mkdir(exist_ok=True)
                    
                    with open(export_path, 'w') as f:
                        f.write("# LEAP Enhanced Analysis Results\n\n")
                        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Filters: Score≥{args.min_score}, IV≤{args.max_iv_percentile}%, Delta≈{args.target_delta}\n\n")
                        
                        for symbol, result in approved_symbols:
                            f.write(f"## {symbol}\n")
                            f.write(f"- **Score**: {result['score']:.3f}\n")
                            f.write(f"- **Recommendation**: {result['recommendation']}\n")
                            
                            if result['option_recommendation'].get('success'):
                                opt = result['option_recommendation']['recommendation']
                                contract = opt['contract']
                                f.write(f"- **Contract**: ${contract.strike} Call {contract.expiration.strftime('%b %Y')}\n")
                                f.write(f"- **Delta**: {contract.delta:.2f}\n")
                                f.write(f"- **Risk Tier**: {opt['risk_assessment']['risk_tier']}\n")
                            f.write("\n")
                    
                    print(f"\n📄 Results exported to {export_path}")
            
            # Stress testing if requested
            if args.stress_test and approved_symbols:
                from backtesting_engine import BacktestEngine
                symbols_to_test = [s for s, _ in approved_symbols]
                
                print(f"\n� Running stress scenarios on {len(symbols_to_test)} symbols...")
                engine = BacktestEngine(None, None)  # Dummy initialization
                stress_results = engine.simulate_stress_scenarios(symbols_to_test)
                
                if 'portfolio_stress' in stress_results:
                    portfolio = stress_results['portfolio_stress']
                    risk_assessment = portfolio.get('risk_assessment', {})
                    
                    print(f"\n📊 STRESS TEST RESULTS:")
                    print(f"Worst Case Loss: {risk_assessment.get('worst_case_option_loss', 0):.1f}%")
                    print(f"Risk Level: {risk_assessment.get('risk_level', 'UNKNOWN')}")
                    print(f"Diversification Benefit: {risk_assessment.get('diversification_benefit', 0):.1f}%")
        
        # Backtesting mode (existing functionality)
        elif args.backtest:
            from backtesting_engine import BacktestEngine
            from datetime import datetime

            print("\n📊 Starting LEAP Options Backtesting")
            print("="*60)

            # Parse dates
            try:
                start_date = datetime.strptime(args.backtest_start, "%Y-%m-%d")
                end_date = datetime.strptime(args.backtest_end, "%Y-%m-%d")
            except ValueError as e:
                print(f"❌ Invalid date format: {e}")
                return

            symbols = [s.strip().upper() for s in args.backtest_symbols.split(',')]

            print(f"� Period: {start_date.date()} to {end_date.date()}")
            print(f"💰 Capital: ${args.backtest_capital:,.0f}")
            print(f"� Symbols: {', '.join(symbols)}")

            engine = BacktestEngine(
                start_date=start_date,
                end_date=end_date,
                initial_capital=args.backtest_capital
            )

            result = engine.run_backtest(symbols)

            # Display results (existing code)
            print("\n📊 BACKTEST RESULTS")
            print(f"Total Return: {result.total_return:+.1f}%")
            print(f"Annualized Return: {result.annualized_return:+.1f}%")
            print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {result.max_drawdown:.1f}%")
            print(f"Win Rate: {result.win_rate:.1f}%")
        
        # Single ticker mode
        elif args.ticker:
            print(f"\n🎯 Analyzing {args.ticker.upper()}")
            
            market_data = client.fetch_market_data(args.ticker.upper())
            if market_data:
                score_result = calculate_weighted_score(args.ticker.upper(), {
                    'iv_percentile': market_data.implied_volatility_rank or 50
                })
                
                option_rec = recommend_leap_structure(
                    args.ticker.upper(),
                    target_delta=args.target_delta,
                    max_iv_percentile=args.max_iv_percentile,
                    risk_tolerance=args.risk_tolerance
                )
                
                analysis_result = {
                    'score': score_result['score'],
                    'recommendation': score_result['recommendation'],
                    'option_recommendation': option_rec,
                    'market_data': market_data
                }
                
                print_enhanced_results(args.ticker.upper(), analysis_result, args)
                
                if args.verbose:
                    print(f"\n📋 Score Breakdown:")
                    for component, data in score_result.get('components', {}).items():
                        print(f"  {component}: {data['score']:.3f} (weight: {data['weight']:.2f}) - {data['description']}")
            else:
                print(f"❌ No data available for {args.ticker}")
        
        else:
            print("Please specify --symbols, --ticker, or --backtest")
            print("Use --help for detailed usage information")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
