"""
Main Entry Point
Deep Agent for Commodity Price Analysis
"""
import asyncio
from pathlib import Path
from config import Config
from agent import CommodityAnalysisAgent
import warnings
warnings.filterwarnings('ignore')

class CommodityAnalysisRunner:
    """Main runner for commodity analysis"""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.data_dir = self.output_dir / "data"
        self.reports_dir = self.output_dir / "reports"
        self.charts_dir = self.output_dir / "charts"
        
        self.data_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        self.charts_dir.mkdir(exist_ok=True)
        
        # Initialize agent
        self.agent = CommodityAnalysisAgent(llm_model="gpt-4")
        
    async def run_analysis(self, user_query: str = None) -> dict:
        """Run the complete commodity analysis"""
        
        print("=" * 80)
        print("COMMODITY ANALYSIS DEEP AGENT")
        print("=" * 80)
        print(f"Analysis Period: {Config.START_DATE} to {Config.HISTORICAL_END}")
        print(f"Forecast Horizon: {Config.FORECAST_DAYS} days")
        print(f"Output Directory: {self.output_dir}")
        print("=" * 80)
        
        # Default query if none provided
        if not user_query:
            user_query = """
            Perform comprehensive commodity analysis for Gold, Aluminum, Platinum, 
            Natural Gas, Gasoline, and Crude Oil. Include statistical analysis, 
            trend identification, pattern detection, correlation analysis, 
            price forecasting, and risk assessment.
            """
        
        print("\nStarting analysis...")
        print("-" * 40)
        
        # Run the analysis
        result = await self.agent.run_analysis(user_query)
        
        if result['success']:
            print("\n✓ Analysis completed successfully!")
            print("\nKey Insights:")
            for i, insight in enumerate(result['key_insights'][:10], 1):
                print(f"  {i}. {insight}")
            
            print(f"\nReports and visualizations saved to: {self.output_dir}")
            
        else:
            print("\n✗ Analysis encountered errors:")
            for error in result['errors']:
                print(f"  - {error}")
        
        return result
    
    async def run_specific_commodities(self, commodities: list, query: str = None):
        """Run analysis for specific commodities"""
        
        if not query:
            commodity_list = ", ".join(commodities)
            query = f"Analyze {commodity_list} with full statistical analysis, forecasting, and risk assessment."
        
        print(f"\nAnalyzing: {', '.join(commodities)}")
        return await self.run_analysis(query)
    
    async def run_category_analysis(self, category: str):
        """Run analysis for a specific category (Metals or Energy)"""
        
        commodities = Config.get_category_commodities(category)
        
        if not commodities:
            print(f"✗ No commodities found for category: {category}")
            return None
        
        query = f"Analyze {category} commodities: {', '.join(commodities)}. " \
                f"Focus on sector-specific trends and inter-commodity relationships."
        
        print(f"\n{category} Category Analysis")
        print("-" * 40)
        print(f"Commodities to analyze: {', '.join(commodities)}\n")
        
        return await self.agent.run_analysis(query, commodities=commodities)

async def main():
    """Main function"""
    
    # Validate configuration
    print("\n🔧 Validating Configuration...")
    if not Config.validate_config():
        print("\n❌ Configuration validation failed!")
        print("\nPlease create a .env file with the following keys:")
        print("  FRED_API_KEY=your_fred_api_key")
        print("  OPENAI_API_KEY=your_openai_api_key")
        print("\nYou can get a FREE FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("You can get an OpenAI API key from: https://platform.openai.com/api-keys")
        return
    
    print("✓ Configuration validated successfully!\n")
    
    # Initialize runner
    runner = CommodityAnalysisRunner()
    
    # Display menu
    print("\n" + "=" * 80)
    print("COMMODITY ANALYSIS OPTIONS")
    print("=" * 80)
    print("\n1. Full Analysis (All 6 commodities)")
    print("2. Metals Analysis (Gold, Aluminium, Platinum)")
    print("3. Energy Analysis (Natural Gas, Gasoline, Crude Oil)")
    print("4. Custom Selection")
    print("5. Quick Test (Gold and Crude Oil only)")
    print("\nEnter choice (1-5) or press Enter for Full Analysis: ", end="")
    
    try:
        choice = input().strip()
        
        if choice == "" or choice == "1":
            # Full analysis
            await runner.run_analysis()
        
        elif choice == "2":
            # Metals only
            await runner.run_category_analysis("Metals")
        
        elif choice == "3":
            # Energy only
            await runner.run_category_analysis("Energy")
        
        elif choice == "4":
            # Custom selection
            print("\nAvailable commodities:")
            for i, commodity in enumerate(Config.get_commodity_list(), 1):
                print(f"  {i}. {commodity}")
            print("\nEnter comma-separated numbers (e.g., 1,3,6): ", end="")
            
            selections = input().strip().split(',')
            commodities = [Config.get_commodity_list()[int(s.strip())-1] for s in selections if s.strip().isdigit()]
            
            if commodities:
                await runner.run_specific_commodities(commodities)
            else:
                print("✗ Invalid selection")
        
        elif choice == "5":
            # Quick test
            print("\n🚀 Running Quick Test...")
            await runner.run_specific_commodities(["Gold", "Crude_Oil"])
        
        else:
            print("✗ Invalid choice. Running full analysis...")
            await runner.run_analysis()
    
    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS SESSION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())