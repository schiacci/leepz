"""
LEAP Strategic Asset Engine - Main Entry Point

Usage:
    # Run full pipeline (discover and analyze)
    python main.py

    # Analyze a specific ticker
    python main.py --ticker NVDA --narrative "AI infrastructure play"
    
    # Analyze multiple tickers
    python main.py --symbols "NVDA,MSFT,TSLA" --narrative "Tech sector analysis"
    
    # Export approved trades
    python main.py --export
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator
from config import config


def setup_logging():
    """Setup logging configuration"""
    log_dir = config.logging.file.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # For now, using print statements
    # In production, configure loguru here
    print(f"📋 Logging to: {config.logging.file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="LEAP Strategic Asset Engine - Discover and analyze LEAP options opportunities"
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of tickers to analyze (e.g., 'NVDA,MSFT,TSLA')"
    )
    
    parser.add_argument(
        "--narrative",
        type=str,
        help="Investment thesis/narrative for manual ticker analysis"
    )
    
    parser.add_argument(
        "--ticker",
        type=str,
        help="Single ticker symbol to analyze (e.g., 'NVDA')"
    )
    
    parser.add_argument(
        "--tickers-per-sector",
        type=int,
        default=2,
        help="Number of tickers per sector (default: 2)"
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
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    
    # Initialize orchestrator
    if args.backtest:
        from datetime import datetime
        start_date = datetime.strptime(args.backtest_start, "%Y-%m-%d")
        orchestrator = Orchestrator(
            backtesting_mode=True,
            backtesting_date=start_date  # Use start date for initial temporal context
        )
    else:
        orchestrator = Orchestrator()

    try:
        # Mode 1: Export approved trades
        if args.export:
            print("\n📤 Exporting approved trades...")
            files = orchestrator.export_approved_trades()
            print(f"✅ Exported {len(files)} trade cards")
            return

        # Mode 2: Backtesting
        if args.backtest:
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
                print("Use YYYY-MM-DD format (e.g., 2020-01-01)")
                return

            # Parse symbols
            symbols = [s.strip().upper() for s in args.backtest_symbols.split(',')]

            print(f"📈 Backtest Period: {start_date.date()} to {end_date.date()}")
            print(f"💰 Initial Capital: ${args.backtest_capital:,.0f}")
            print(f"📊 Universe: {', '.join(symbols)}")
            print(f"🎯 Strategy: LEAP Call Options (1-2 year expirations)")
            print("="*60)

            # Initialize backtesting engine
            engine = BacktestEngine(
                start_date=start_date,
                end_date=end_date,
                initial_capital=args.backtest_capital,
                max_positions=5,
                position_size_pct=0.20
            )

            try:
                # Run backtest
                result = engine.run_backtest(symbols)

                # Display comprehensive results
                print("\n" + "="*60)
                print("📊 BACKTEST RESULTS")
                print("="*60)
                print(f"💰 Final Portfolio Value: ${result.total_return + args.backtest_capital:,.0f}")
                print(f"📈 Total Return: {result.total_return:+.1f}%")
                print(f"🎯 Annualized Return: {result.annualized_return:+.1f}%")
                print(f"⚡ Sharpe Ratio: {result.sharpe_ratio:.2f}")
                print(f"📉 Maximum Drawdown: {result.max_drawdown:.1f}%")
                print(f"🏆 Win Rate: {result.win_rate:.1f}%")
                print(f"📊 Total Trades: {result.total_trades}")
                print(f"💪 Profitable Trades: {result.profitable_trades}")
                print(f"📊 Average Trade Return: ${result.avg_trade_return:+.2f}")

                # Special handling for no trades scenario
                if result.total_trades == 0:
                    print("\n" + "⚠️  NO TRADES EXECUTED" + " ⚠️")
                    print("This could be due to:")
                    print("• Strategy conditions not met (technical indicators)")
                    print("• Errors during opportunity scanning")
                    print("• Insufficient capital or position limits")
                    print("• Data availability issues")
                    print("\n💡 Check the logs above for 'Error finding opportunities' messages")

                # Risk assessment
                if result.total_trades == 0:
                    risk_assessment = "⚠️ INSUFFICIENT DATA - No trades executed to evaluate"
                elif result.sharpe_ratio > 1.5:
                    risk_assessment = "🟢 EXCELLENT - Strong risk-adjusted returns"
                elif result.sharpe_ratio > 1.0:
                    risk_assessment = "🟡 GOOD - Decent risk-adjusted performance"
                elif result.sharpe_ratio > 0.5:
                    risk_assessment = "🟠 FAIR - Acceptable but could be improved"
                else:
                    risk_assessment = "🔴 POOR - High risk, low reward"

                print(f"🎖️ Risk Assessment: {risk_assessment}")

                print("\n" + "-"*60)
                print("💡 Professional Interpretation:")
                if result.total_trades > 0:
                    print("• Sharpe > 1.5: Excellent risk-adjusted returns")
                    print("• Max Drawdown < 20%: Reasonable portfolio stress")
                    print("• Win Rate > 50%: Strategy has edge")
                    print("• Annualized > 15%: Strong performance potential")
                else:
                    print("• No trades executed - strategy needs debugging")
                    print("• Check date ranges and market conditions")
                    print("• Verify technical indicators are working")
                    print("• Ensure sufficient historical data available")
                print("-"*60)

                # Ask about detailed analysis
                response = input("\n🔍 Show detailed trade log? (y/n): ").strip().lower()
                if response == 'y':
                    print("\n📋 TRADE LOG")
                    print("="*80)
                    print(f"\n📋 TRADE LOG (All {len(result.trades_executed)} Trades)")
                    print("=" * 80)
                    print(f"{'Date':<12} | {'Action':<6} | {'Contract':<40} | {'Expiration':<12} | {'Details'}")
                    print("-" * 80)
                    
                    for trade in result.trades_executed:  # All trades
                        # Format contract info based on trade type
                        if trade['action'] == 'EXPIRE':
                            contract_info = f"{trade['symbol']} ${trade['strike']:.2f} Call"
                            details = f"PNL: ${trade.get('pnl', 0):+,.2f}"
                        else:
                            contract_info = (f"{trade.get('contracts', 'N/A')}x {trade['symbol']} "
                                          f"${trade['strike']:.2f} Call")
                            cost = trade.get('total_cost', 0)
                            stock_price = trade.get('stock_price', 0)
                            details = (f"Cost: ${cost:,.2f} | "
                                     f"Stock: ${stock_price:.2f}")
                        
                        # Get expiration date (either from the trade or its expiration field)
                        expiration = trade.get('expiration_date', 
                                            trade.get('expiration', 'N/A'))
                        if hasattr(expiration, 'strftime'):  # If it's a datetime object
                            expiration = expiration.strftime('%Y-%m-%d')
                        
                        print(f"{trade['date'].strftime('%Y-%m-%d')} | "
                              f"{trade['action']:<6} | "
                              f"{contract_info:<40} | "
                              f"{expiration:<12} | "
                              f"{details}")

                return

            except Exception as e:
                print(f"❌ Backtest failed: {e}")
                import traceback
                traceback.print_exc()
                return

        # Mode 3: Batch symbol analysis
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
            print(f"\n📊 Batch Mode: Analyzing {len(symbols)} symbols")
            print(f"Symbols: {', '.join(symbols)}")
            
            successful_analyses = []
            
            for i, symbol in enumerate(symbols, 1):
                print(f"\n{'='*60}")
                print(f"📈 Analyzing {symbol} ({i}/{len(symbols)})")
                print(f"{'='*60}")
                
                trade_card = orchestrator.analyze_single_ticker(
                    ticker=symbol,
                    narrative=args.narrative
                )
                
                if trade_card:
                    successful_analyses.append(trade_card)
                    print("\n" + "-"*60)
                    print(f"✅ {symbol} Analysis Complete")
                    print("-"*60)
                    
                    # Add clear BUY/SELL recommendation
                    if trade_card.approved:
                        print(f"🎯 RECOMMENDATION: BUY {symbol} LEAP Call")
                        print(f"   Entry: ${trade_card.entry_price:.2f} | Target: ${trade_card.projected_exit_price_1yr:.2f}")
                        print(f"   Expected Return: {trade_card.projected_return_pct:+.1f}%")
                    else:
                        print(f"❌ RECOMMENDATION: SELL/AVOID {symbol} LEAP Call")
                        print(f"   Risk concerns outweigh potential rewards")
                    
                    print("-"*60)
                    print(trade_card.to_markdown())
                else:
                    print(f"\n❌ Analysis failed for {symbol}")
            
            # Summary
            if successful_analyses:
                print("\n" + "="*60)
                print(f"📊 BATCH ANALYSIS COMPLETE - {len(successful_analyses)}/{len(symbols)} successful")
                print("="*60)
                
                # Ask if user wants to export
                response = input("📤 Export successful analyses to markdown? (y/n): ").strip().lower()
                if response == 'y':
                    # Export each successful analysis
                    for card in successful_analyses:
                        markdown_path = Path(f"./outputs/{card.ticker}_trade_card.md")
                        markdown_path.parent.mkdir(exist_ok=True)
                        markdown_path.write_text(card.to_markdown())
                        
                        # Add recommendation status to export message
                        rec_status = "BUY" if card.approved else "SELL/AVOID"
                        print(f"📄 Exported {card.ticker} to {markdown_path} - {rec_status} {card.ticker} LEAP Call")
            else:
                print("\n❌ No analyses completed successfully")
            
            return
        
        # Mode 3: Manual single ticker analysis
        if args.ticker:
            print(f"\n🎯 Manual Mode: Analyzing {args.ticker}")
            trade_card = orchestrator.analyze_single_ticker(
                ticker=args.ticker.upper(),
                narrative=args.narrative
            )
            
            if trade_card:
                print("\n" + "="*60)
                print("📄 Trade Card Generated")
                print("="*60)
                print(trade_card.to_markdown())
            else:
                print(f"\n❌ Analysis failed for {args.ticker}")
            return
        
        # Mode 3: Full pipeline (discovery + analysis)
        print("\n🚀 Running Full LEAP Discovery Pipeline")
        trade_cards = orchestrator.run_full_pipeline(
            num_sectors=args.sectors,
            tickers_per_sector=args.tickers_per_sector
        )
        
        # Print approved recommendations
        approved = [tc for tc in trade_cards if tc.approved]
        
        if approved:
            print("\n" + "="*60)
            print(f"✅ APPROVED RECOMMENDATIONS ({len(approved)})")
            print("="*60)
            
            for card in approved:
                print(f"\n{card.ticker}: ${card.entry_price:.2f} → ${card.projected_exit_price_1yr:.2f} ({card.projected_return_pct:+.1f}%)")
                print(f"  Contract: ${card.contract.strike} Call {card.contract.expiration.strftime('%Y-%m-%d')}")
                print(f"  Delta: {card.contract.delta:.3f} | Risk Score: {card.risk_score:.2f}")
        else:
            print("\n⚠️ No trades approved by risk criteria")
        
        # Ask if user wants to export
        if approved:
            print("\n" + "="*60)
            response = input("📤 Export approved trades to markdown? (y/n): ").strip().lower()
            if response == 'y':
                orchestrator.export_approved_trades()
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
