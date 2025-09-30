"""
Configuration file for Commodity Analysis Deep Agent
"""
import os
from datetime import datetime, timedelta
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the commodity analysis system"""
    
    # API Keys
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Time Period Configuration (2020-2030 focus)
    START_DATE = "2020-01-01"
    HISTORICAL_END = datetime.now().strftime("%Y-%m-%d")
    FORECAST_END = "2030-12-31"
    
    # Calculate forecast days from today to 2030
    today = datetime.now()
    end_2030 = datetime(2030, 12, 31)
    FORECAST_DAYS = (end_2030 - today).days
    
    # Commodity FRED Tickers
    COMMODITY_TICKERS: Dict[str, Dict[str, str]] = {
        # Metals
        "Gold": {
            "fred_ticker": "WPUFD49207",  # PPI: Gold Ores
            "category": "Metals",
            "unit": "Index"
        },
        "Aluminium": {
            "fred_ticker": "PALUMUSDM",  # LME Aluminum price
            "category": "Metals",
            "unit": "USD/Metric Ton"
        },
        "Platinum": {
            "fred_ticker": "WPUFD49502",  # PPI: Platinum Group Metals
            "category": "Metals",
            "unit": "Index"
        },
        # Energy
        "Natural_Gas": {
            "fred_ticker": "DHHNGSP",  # Henry Hub Natural Gas Spot Price
            "category": "Energy",
            "unit": "USD/Million BTU"
        },
        "Gasoline": {
            "fred_ticker": "GASREGW",  # US Regular Gas Price
            "category": "Energy",
            "unit": "USD/Gallon"
        },
        "Crude_Oil": {
            "fred_ticker": "DCOILWTICO",  # WTI Crude Oil
            "category": "Energy",
            "unit": "USD/Barrel"
        }
    }
    
    # Analysis Parameters
    VOLATILITY_WINDOW = 30  # days for volatility calculation
    TREND_WINDOW = 90  # days for trend analysis
    CORRELATION_WINDOW = 252  # trading days (1 year)
    
    # Forecasting Parameters
    PROPHET_CHANGEPOINT_PRIOR = 0.05
    PROPHET_SEASONALITY_PRIOR = 10
    PROPHET_HOLIDAYS_PRIOR = 10
    
    # Risk Analysis Parameters
    VAR_CONFIDENCE_LEVEL = 0.95  # 95% confidence for VaR
    CVAR_CONFIDENCE_LEVEL = 0.95  # 95% confidence for CVaR
    
    # LLM Configuration
    LLM_MODEL = "gpt-4"
    LLM_TEMPERATURE = 0.0
    MAX_TOKENS = 4096
    
    # Output Configuration
    OUTPUT_DIR = "./results"
    DATA_DIR = "./results/data"
    REPORTS_DIR = "./results/reports"
    CHARTS_DIR = "./results/charts"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present"""
        if not cls.FRED_API_KEY:
            print("WARNING: FRED_API_KEY not found in environment variables")
            print("Please set FRED_API_KEY in your .env file")
            return False
        
        if not cls.OPENAI_API_KEY:
            print("WARNING: OPENAI_API_KEY not found in environment variables")
            print("Please set OPENAI_API_KEY in your .env file")
            return False
            
        return True
    
    @classmethod
    def get_commodity_list(cls) -> List[str]:
        """Get list of all commodity names"""
        return list(cls.COMMODITY_TICKERS.keys())
    
    @classmethod
    def get_category_commodities(cls, category: str) -> List[str]:
        """Get commodities by category (Metals or Energy)"""
        return [
            name for name, info in cls.COMMODITY_TICKERS.items()
            if info["category"] == category
        ]
    
    @classmethod
    def get_ticker(cls, commodity: str) -> str:
        """Get FRED ticker for a commodity"""
        return cls.COMMODITY_TICKERS.get(commodity, {}).get("fred_ticker", "")

# Validate configuration on import
if __name__ == "__main__":
    if Config.validate_config():
        print("✓ Configuration validated successfully")
        print(f"\nCommodities configured:")
        for commodity, info in Config.COMMODITY_TICKERS.items():
            print(f"  - {commodity}: {info['fred_ticker']} ({info['category']})")
        print(f"\nAnalysis period: {Config.START_DATE} to {Config.HISTORICAL_END}")
        print(f"Forecast horizon: {Config.FORECAST_DAYS} days (until {Config.FORECAST_END})")
    else:
        print("✗ Configuration validation failed")