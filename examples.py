"""
Example usage of LEAP Strategic Asset Engine

This script demonstrates how to use the engine programmatically
instead of via the command line.
"""
from pathlib import Path
from orchestrator import Orchestrator
from config import config


def example_full_pipeline():
    """Example: Run full discovery pipeline"""
    print("Example 1: Full Pipeline\n" + "="*60)
    
    # Initialize
    orchestrator = Orchestrator()
    
    # Run pipeline
    trade_cards = orchestrator.run_full_pipeline(
        num_sectors=3,
        tickers_per_sector=2
    )
    
    # Print approved recommendations
    approved = [tc for tc in trade_cards if tc.approved]
    print(f"\n✅ Found {len(approved)} approved opportunities")
    
    return approved


def example_single_ticker():
    """Example: Analyze a specific ticker"""
    print("\n\nExample 2: Single Ticker Analysis\n" + "="*60)
    
    orchestrator = Orchestrator()
    
    # Analyze AAPL
    trade_card = orchestrator.analyze_single_ticker(
        ticker="AAPL",
        narrative="Premium consumer brand with services growth and potential Vision Pro catalyst"
    )
    
    if trade_card:
        print("\n📄 Trade Card:")
        print(trade_card.to_markdown())
        return trade_card
    else:
        print("❌ No suitable LEAP found")
        return None


def example_database_queries():
    """Example: Query the database"""
    print("\n\nExample 3: Database Queries\n" + "="*60)
    
    from database.db import Database
    
    db = Database(config.database.path)
    
    # Get recent approved trades
    print("\n📊 Recent Approved Trades:")
    approved_trades = db.get_recent_trade_cards(limit=5, approved_only=True)
    
    for trade in approved_trades:
        print(f"  - {trade['ticker']}: {trade['projected_return_pct']:.1f}% return")
    
    # Get ticker history
    if approved_trades:
        ticker = approved_trades[0]['ticker']
        print(f"\n📈 History for {ticker}:")
        history = db.get_ticker_history(ticker)
        print(f"  Total recommendations: {len(history)}")


def example_custom_workflow():
    """Example: Build a custom workflow"""
    print("\n\nExample 4: Custom Workflow\n" + "="*60)
    
    from agents.ai_agents import DiscoveryScout
    from data.market_data_client import MarketDataClient
    
    # Step 1: Manual discovery
    scout = DiscoveryScout()
    print("\n🔍 Discovering trends...")
    
    # You could also manually create trending assets instead of using AI
    from models import TrendingAsset
    
    manual_assets = [
        TrendingAsset(
            ticker="MSFT",
            sector="Cloud Computing",
            narrative="Azure cloud growth + AI integration via OpenAI partnership",
            confidence_score=0.80,
            source="Manual"
        ),
        TrendingAsset(
            ticker="GOOGL",
            sector="AI/Search",
            narrative="Gemini AI rollout + search monopoly + cloud growth",
            confidence_score=0.75,
            source="Manual"
        )
    ]
    
    # Step 2: Fetch data
    data_client = MarketDataClient()
    
    for asset in manual_assets:
        print(f"\n📊 Fetching {asset.ticker}...")
        market_data = data_client.fetch_market_data(asset.ticker)
        
        if market_data and market_data.option_contracts:
            print(f"  ✅ Found {len(market_data.option_contracts)} LEAP contracts")
            best = market_data.option_contracts[0]
            print(f"  Best: ${best.strike} Call {best.expiration.strftime('%Y-%m-%d')}")
            print(f"  Delta: {best.delta:.3f}, IV: {best.implied_volatility:.2%}")


def main():
    """Run all examples"""
    print("🚀 LEAP Strategic Asset Engine - Examples\n")
    
    try:
        # Uncomment the examples you want to run:
        
        # example_full_pipeline()
        # example_single_ticker()
        # example_database_queries()
        example_custom_workflow()
        
        print("\n" + "="*60)
        print("✅ Examples completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
