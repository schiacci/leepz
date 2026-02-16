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
        "--sectors",
        type=int,
        default=5,
        help="Number of sectors to discover (default: 5)"
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
        "--db-path",
        type=str,
        help="Custom database path (default: ./database/leap_engine.db)"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    
    # Initialize orchestrator
    db_path = Path(args.db_path) if args.db_path else None
    orchestrator = Orchestrator(db_path=db_path)
    
    try:
        # Mode 1: Export approved trades
        if args.export:
            print("\n📤 Exporting approved trades...")
            files = orchestrator.export_approved_trades()
            print(f"✅ Exported {len(files)} trade cards")
            return
        
        # Mode 2: Batch symbol analysis
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
                        print(f"📄 Exported {card.ticker} to {markdown_path}")
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
