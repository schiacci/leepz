"""
Orchestrator for LEAP Strategic Asset Engine
Manages the state and data flow between all components
"""
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from models import TradeCard, TrendingAsset, MarketData, ReasoningOutput, CritiqueOutput
from agents.ai_agents import DiscoveryScout, QuantReasoningEngine, RiskCritic
from data.market_data_client import MarketDataClient
from database.db import Database
from config import config


class Orchestrator:
    """
    Main controller that orchestrates the LEAP discovery and analysis workflow
    
    Workflow:
    1. Discovery: Find trending assets (DiscoveryScout)
    2. Data Fetch: Get market data and option chains (MarketDataClient)
    3. Reasoning: Analyze LEAP opportunity (QuantReasoningEngine)
    4. Critique: Risk assessment (RiskCritic)
    5. Generate: Create Trade Card
    6. Persist: Save to database
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize all components"""
        self.db = Database(db_path or config.database.path)
        self.scout = DiscoveryScout()
        self.data_client = MarketDataClient()
        self.quant = QuantReasoningEngine()
        self.critic = RiskCritic()
        
        print("🚀 LEAP Strategic Asset Engine initialized")
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration before running"""
        try:
            config.validate_api_keys()
            print("✅ Configuration validated")
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            print("💡 Please set OPENROUTER_API_KEY in your .env file")
            raise
    
    def run_full_pipeline(self, num_sectors: int = 5, 
                         tickers_per_sector: int = 2) -> List[TradeCard]:
        """
        Execute the complete LEAP discovery and analysis pipeline
        
        Args:
            num_sectors: Number of sectors to analyze
            tickers_per_sector: Tickers per sector
            
        Returns:
            List of TradeCard objects (approved recommendations)
        """
        print("\n" + "="*60)
        print("🎯 Starting LEAP Strategic Asset Engine Pipeline")
        print("="*60 + "\n")
        
        # Step 1: Discovery
        print("📡 Phase 1: Discovery Scout - Scanning for trending assets...")
        trending_assets = self.scout.discover_trending_assets(
            num_sectors=num_sectors,
            tickers_per_sector=tickers_per_sector
        )
        
        if not trending_assets:
            print("❌ No trending assets discovered. Exiting.")
            return []
        
        # Save discovered assets
        for asset in trending_assets:
            self.db.save_trending_asset(asset)
        
        print(f"✅ Discovered {len(trending_assets)} assets\n")
        
        # Step 2: Analyze each asset
        trade_cards = []
        
        for i, asset in enumerate(trending_assets, 1):
            print(f"\n{'='*60}")
            print(f"📊 Analyzing {i}/{len(trending_assets)}: {asset.ticker}")
            print(f"{'='*60}")
            
            try:
                trade_card = self._analyze_asset(asset)
                
                if trade_card:
                    trade_cards.append(trade_card)
                    
                    # Save to database
                    self.db.save_trade_card(trade_card)
                    
                    # Print summary
                    self._print_trade_summary(trade_card)
                
            except Exception as e:
                print(f"❌ Error analyzing {asset.ticker}: {e}")
                continue
        
        # Final summary
        print("\n" + "="*60)
        print("🎉 Pipeline Complete!")
        print("="*60)
        print(f"Total Analyzed: {len(trending_assets)}")
        print(f"Trade Cards Generated: {len(trade_cards)}")
        print(f"Approved Recommendations: {sum(1 for tc in trade_cards if tc.approved)}")
        print("="*60 + "\n")
        
        return trade_cards
    
    def _analyze_asset(self, asset: TrendingAsset) -> Optional[TradeCard]:
        """
        Analyze a single asset through the full pipeline
        
        Returns:
            TradeCard if successful, None otherwise
        """
        ticker = asset.ticker
        
        # Step 2: Fetch Market Data
        print(f"  📈 Fetching market data for {ticker}...")
        market_data = self.data_client.fetch_market_data(ticker)
        
        if not market_data or not market_data.option_contracts:
            print(f"  ⚠️ No suitable LEAP options found for {ticker}")
            return None
        
        print(f"  ✅ Found {len(market_data.option_contracts)} LEAP candidates")
        
        # Step 3: Quant Reasoning
        print(f"  🧮 Running quantitative analysis...")
        reasoning = self.quant.analyze_leap_opportunity(
            ticker=ticker,
            market_data=market_data,
            narrative=asset.narrative
        )
        
        if not reasoning:
            print(f"  ❌ Quant analysis failed for {ticker}")
            return None
        
        # Log reasoning to database
        self.db.log_reasoning(
            ticker=ticker,
            agent_name="QuantReasoningEngine",
            prompt="LEAP Analysis",
            response=reasoning.chain_of_thought,
            model_used=config.openrouter.deepseek_r1_model
        )
        
        # Step 4: Risk Critique
        print(f"  🛡️ Running risk assessment...")
        earnings_date = self.data_client.get_earnings_date(ticker)
        
        critique = self.critic.critique_recommendation(
            ticker=ticker,
            reasoning=reasoning,
            market_data=market_data,
            earnings_date=earnings_date
        )
        
        # Log critique
        self.db.log_reasoning(
            ticker=ticker,
            agent_name="RiskCritic",
            prompt="Risk Assessment",
            response=critique.critique_reasoning,
            model_used=config.openrouter.deepseek_v3_model
        )
        
        # Step 5: Generate Trade Card
        trade_card = self._generate_trade_card(
            asset=asset,
            market_data=market_data,
            reasoning=reasoning,
            critique=critique
        )
        
        return trade_card
    
    def _generate_trade_card(self, asset: TrendingAsset, market_data: MarketData,
                            reasoning: ReasoningOutput, critique: CritiqueOutput) -> TradeCard:
        """Generate final Trade Card from all analysis"""
        
        contract = reasoning.recommended_contract
        entry_price = (contract.bid + contract.ask) / 2
        
        # Get projected 1-year values
        if reasoning.scenario_analysis:
            projected_exit = reasoning.scenario_analysis.expected_value
            projected_return = reasoning.scenario_analysis.expected_return_pct
        else:
            projected_exit = entry_price
            projected_return = 0.0
        
        trade_card = TradeCard(
            ticker=asset.ticker,
            sector=asset.sector,
            discovery_narrative=asset.narrative,
            contract=contract,
            entry_price=entry_price,
            projected_exit_price_1yr=projected_exit,
            projected_return_pct=projected_return,
            max_loss=entry_price,  # Maximum loss is premium paid
            break_even_price=reasoning.break_even_price or contract.strike,
            risk_score=critique.risk_score,
            scenario_analysis=reasoning.scenario_analysis,
            risk_flags=critique.risk_flags,
            approved=critique.approved,
            quant_reasoning=reasoning.chain_of_thought,
            risk_critique=critique.critique_reasoning
        )
        
        return trade_card
    
    def _print_trade_summary(self, card: TradeCard):
        """Print a compact summary of the trade card"""
        status = "✅ APPROVED" if card.approved else "❌ REJECTED"
        
        print(f"\n  {status}")
        print(f"  Contract: {card.ticker} ${card.contract.strike} Call {card.contract.expiration.strftime('%Y-%m-%d')}")
        print(f"  Entry: ${card.entry_price:.2f} | Projected 1Y: ${card.projected_exit_price_1yr:.2f} ({card.projected_return_pct:+.1f}%)")
        print(f"  Risk Score: {card.risk_score:.2f}/1.0")
        
        if card.risk_flags.critical_issues:
            print(f"  🚨 Critical Issues: {', '.join(card.risk_flags.critical_issues)}")
    
    def analyze_single_ticker(self, ticker: str, narrative: Optional[str] = None) -> Optional[TradeCard]:
        """
        Analyze a single ticker (manual mode)
        
        Args:
            ticker: Stock symbol
            narrative: Optional custom narrative
            
        Returns:
            TradeCard or None
        """
        print(f"\n{'='*60}")
        print(f"🎯 Manual Analysis: {ticker}")
        print(f"{'='*60}\n")
        
        # Create a TrendingAsset object
        asset = TrendingAsset(
            ticker=ticker,
            sector="Manual Entry",
            narrative=narrative or f"Manual analysis of {ticker}",
            confidence_score=1.0,
            source="Manual"
        )
        
        # Save to database
        self.db.save_trending_asset(asset)
        
        # Run analysis
        trade_card = self._analyze_asset(asset)
        
        if trade_card:
            self.db.save_trade_card(trade_card)
            self._print_trade_summary(trade_card)
            
            # Export markdown
            markdown_path = Path(f"./outputs/{ticker}_trade_card.md")
            markdown_path.parent.mkdir(exist_ok=True)
            markdown_path.write_text(trade_card.to_markdown())
            print(f"\n📄 Trade card exported to: {markdown_path}")
        
        return trade_card
    
    def export_approved_trades(self, output_dir: Path = Path("./outputs")) -> List[Path]:
        """
        Export all approved trades as markdown files
        
        Returns:
            List of file paths created
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        recent_cards = self.db.get_recent_trade_cards(limit=100, approved_only=True)
        
        files_created = []
        for card_data in recent_cards:
            ticker = card_data['ticker']
            filepath = output_dir / f"{ticker}_trade_card.md"
            
            # Reconstruct TradeCard (simplified version)
            # In production, you'd want to properly deserialize from JSON
            markdown = f"""# {ticker} Trade Card
            
Created: {card_data['created_at']}
Approved: {'Yes' if card_data['approved'] else 'No'}
Entry Price: ${card_data['entry_price']:.2f}
Projected Return: {card_data['projected_return_pct']:.1f}%
"""
            filepath.write_text(markdown)
            files_created.append(filepath)
        
        print(f"📄 Exported {len(files_created)} approved trade cards to {output_dir}")
        return files_created
