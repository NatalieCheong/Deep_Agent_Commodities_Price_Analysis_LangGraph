"""
Data Collection Module
Handles fetching commodity price data from FRED and Yahoo Finance
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from fredapi import Fred
import yfinance as yf
from config import Config
import warnings
import ssl
import certifi
warnings.filterwarnings('ignore')

# Fix SSL certificate verification for macOS
import urllib.request
ssl_context = ssl.create_default_context(cafile=certifi.where())
urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context)))

class DataCollector:
    """Collects and processes commodity price data"""
    
    def __init__(self, fred_api_key: str = None):
        """
        Initialize data collector
        
        Args:
            fred_api_key: FRED API key (uses Config if not provided)
        """
        self.fred_api_key = fred_api_key or Config.FRED_API_KEY
        
        if not self.fred_api_key:
            raise ValueError("FRED API key is required. Set FRED_API_KEY in .env file")
        
        self.fred = Fred(api_key=self.fred_api_key)
        self.cache = {}
    
    def fetch_commodity_data(self, commodity_name: str, 
                           start_date: str = None,
                           end_date: str = None) -> pd.DataFrame:
        """
        Fetch data for a specific commodity
        
        Args:
            commodity_name: Name of commodity (e.g., 'Gold', 'Crude_Oil')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
        
        Returns:
            DataFrame with date index and 'value' column
        """
        start_date = start_date or Config.START_DATE
        end_date = end_date or Config.HISTORICAL_END
        
        # Check cache
        cache_key = f"{commodity_name}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        # Get ticker from config
        commodity_info = Config.COMMODITY_TICKERS.get(commodity_name)
        if not commodity_info:
            raise ValueError(f"Unknown commodity: {commodity_name}")
        
        fred_ticker = commodity_info['fred_ticker']
        
        try:
            # Fetch from FRED
            data = self.fred.get_series(
                fred_ticker,
                observation_start=start_date,
                observation_end=end_date
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['value'])
            df.index.name = 'date'
            df = df.reset_index()
            
            # Clean data
            df = self._clean_data(df)
            
            # Set date as index
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.sort_index()
            
            # Cache the result
            self.cache[cache_key] = df.copy()
            
            return df
            
        except Exception as e:
            print(f"Error fetching {commodity_name} from FRED: {e}")
            # Try Yahoo Finance as fallback
            return self._fetch_from_yahoo(commodity_name, start_date, end_date)
    
    def fetch_all_commodities(self, start_date: str = None, 
                            end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all configured commodities
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
        
        Returns:
            Dictionary mapping commodity names to DataFrames
        """
        start_date = start_date or Config.START_DATE
        end_date = end_date or Config.HISTORICAL_END
        
        all_data = {}
        
        for commodity_name in Config.get_commodity_list():
            try:
                print(f"Fetching {commodity_name}...")
                data = self.fetch_commodity_data(commodity_name, start_date, end_date)
                
                if not data.empty:
                    all_data[commodity_name] = data
                    print(f"  ✓ Fetched {len(data)} records for {commodity_name}")
                else:
                    print(f"  ✗ No data available for {commodity_name}")
                    
            except Exception as e:
                print(f"  ✗ Error fetching {commodity_name}: {e}")
        
        return all_data
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data
        
        Args:
            df: Raw DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        # Remove NaN values
        df = df.dropna(subset=['value'])
        
        # Remove negative prices (invalid)
        df = df[df['value'] > 0]
        
        # Remove outliers (values more than 5 std devs from mean)
        mean = df['value'].mean()
        std = df['value'].std()
        df = df[np.abs(df['value'] - mean) <= 5 * std]
        
        return df
    
    def _fetch_from_yahoo(self, commodity_name: str,
                         start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fallback method to fetch from Yahoo Finance
        
        Args:
            commodity_name: Name of commodity
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with commodity data
        """
        # Yahoo Finance ticker mapping
        yahoo_tickers = {
            'Gold': 'GC=F',
            'Crude_Oil': 'CL=F',
            'Natural_Gas': 'NG=F',
            'Gasoline': 'RB=F'
        }
        
        ticker = yahoo_tickers.get(commodity_name)
        if not ticker:
            return pd.DataFrame()
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                return pd.DataFrame()
            
            # Use Close price
            df = pd.DataFrame({
                'value': data['Close']
            })
            
            df = self._clean_data(df.reset_index())
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            return df
            
        except Exception as e:
            print(f"Yahoo Finance fallback failed for {commodity_name}: {e}")
            return pd.DataFrame()
    
    def get_data_quality_report(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Generate data quality report for all commodities
        
        Args:
            data_dict: Dictionary of commodity DataFrames
        
        Returns:
            Dictionary with quality metrics for each commodity
        """
        report = {}
        
        for commodity, df in data_dict.items():
            if df.empty:
                report[commodity] = {"status": "no_data"}
                continue
            
            # Calculate metrics
            total_days = (df.index.max() - df.index.min()).days
            actual_records = len(df)
            expected_records = total_days  # Approximate
            
            completeness = (actual_records / expected_records) * 100 if expected_records > 0 else 0
            
            report[commodity] = {
                "status": "ok",
                "records": actual_records,
                "date_range": {
                    "start": df.index.min().strftime("%Y-%m-%d"),
                    "end": df.index.max().strftime("%Y-%m-%d"),
                    "days": total_days
                },
                "completeness": round(completeness, 2),
                "missing_days": max(0, expected_records - actual_records),
                "price_range": {
                    "min": float(df['value'].min()),
                    "max": float(df['value'].max()),
                    "mean": float(df['value'].mean()),
                    "std": float(df['value'].std())
                },
                "latest_price": float(df['value'].iloc[-1]),
                "latest_date": df.index[-1].strftime("%Y-%m-%d")
            }
        
        return report
    
    def resample_data(self, df: pd.DataFrame, frequency: str = 'D') -> pd.DataFrame:
        """
        Resample data to different frequency
        
        Args:
            df: DataFrame with date index
            frequency: Pandas frequency string ('D', 'W', 'M', etc.)
        
        Returns:
            Resampled DataFrame
        """
        return df.resample(frequency).mean().dropna()
    
    def align_dates(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align all commodities to same date range
        
        Args:
            data_dict: Dictionary of commodity DataFrames
        
        Returns:
            Dictionary with aligned DataFrames
        """
        if not data_dict:
            return {}
        
        # Find common date range
        min_date = max(df.index.min() for df in data_dict.values() if not df.empty)
        max_date = min(df.index.max() for df in data_dict.values() if not df.empty)
        
        # Filter all to common range
        aligned_data = {}
        for commodity, df in data_dict.items():
            if not df.empty:
                aligned_data[commodity] = df[(df.index >= min_date) & (df.index <= max_date)]
        
        return aligned_data

# Standalone test function
if __name__ == "__main__":
    print("Testing Data Collector...")
    print("=" * 60)
    
    # Initialize
    collector = DataCollector()
    
    # Test single commodity
    print("\n1. Testing single commodity fetch (Gold)...")
    gold_data = collector.fetch_commodity_data("Gold")
    print(f"   Records: {len(gold_data)}")
    print(f"   Date range: {gold_data.index.min()} to {gold_data.index.max()}")
    print(f"   Latest price: ${gold_data['value'].iloc[-1]:.2f}")
    
    # Test all commodities
    print("\n2. Testing all commodities fetch...")
    all_data = collector.fetch_all_commodities()
    print(f"   Total commodities fetched: {len(all_data)}")
    
    # Data quality report
    print("\n3. Data Quality Report:")
    quality = collector.get_data_quality_report(all_data)
    for commodity, metrics in quality.items():
        if metrics.get("status") == "ok":
            print(f"\n   {commodity}:")
            print(f"     Records: {metrics['records']}")
            print(f"     Completeness: {metrics['completeness']:.1f}%")
            print(f"     Latest: ${metrics['latest_price']:.2f} ({metrics['latest_date']})")
    
    print("\n" + "=" * 60)
    print("✓ Data Collector test complete")