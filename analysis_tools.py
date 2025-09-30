"""
Analysis Tools Module
Statistical analysis, trend detection, and pattern recognition
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class CommodityAnalyzer:
    """Performs various analyses on commodity price data"""
    
    def __init__(self):
        self.results = {}
    
    def statistical_summary(self, data: pd.DataFrame, commodity_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive statistical summary
        
        Args:
            data: DataFrame with 'value' column
            commodity_name: Name of the commodity
        
        Returns:
            Dictionary with statistical metrics
        """
        if data.empty:
            return {"error": f"No data for {commodity_name}"}
        
        values = data['value'].values
        
        summary = {
            "commodity": commodity_name,
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
            "cv": float(np.std(values) / np.mean(values) * 100),  # Coefficient of variation
            "skewness": float(stats.skew(values)),
            "kurtosis": float(stats.kurtosis(values)),
            "quartiles": {
                "q1": float(np.percentile(values, 25)),
                "q2": float(np.percentile(values, 50)),
                "q3": float(np.percentile(values, 75))
            }
        }
        
        # Recent statistics (last 90 days)
        if len(data) >= 90:
            recent_values = values[-90:]
            summary["recent_90d"] = {
                "mean": float(np.mean(recent_values)),
                "std": float(np.std(recent_values)),
                "min": float(np.min(recent_values)),
                "max": float(np.max(recent_values)),
                "change_pct": float((values[-1] - values[-90]) / values[-90] * 100)
            }
        
        return summary
    
    def trend_analysis(self, data: pd.DataFrame, commodity_name: str, 
                      window: int = 90) -> Dict[str, Any]:
        """
        Analyze price trends using various methods
        
        Args:
            data: DataFrame with 'value' column
            commodity_name: Name of the commodity
            window: Window size for moving averages
        
        Returns:
            Dictionary with trend information
        """
        if data.empty or len(data) < window:
            return {"error": f"Insufficient data for trend analysis of {commodity_name}"}
        
        df = data.copy()
        values = df['value'].values
        
        # Moving averages
        df['ma_short'] = df['value'].rolling(window=30).mean()
        df['ma_long'] = df['value'].rolling(window=window).mean()
        
        # Linear regression trend
        X = np.arange(len(values)).reshape(-1, 1)
        y = values
        
        model = LinearRegression()
        model.fit(X, y)
        trend_line = model.predict(X)
        
        # Trend direction
        current_ma_short = df['ma_short'].iloc[-1]
        current_ma_long = df['ma_long'].iloc[-1]
        
        if current_ma_short > current_ma_long:
            trend_direction = "upward"
        elif current_ma_short < current_ma_long:
            trend_direction = "downward"
        else:
            trend_direction = "neutral"
        
        # Trend strength (R-squared)
        ss_res = np.sum((y - trend_line) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Price momentum
        momentum = {}
        if len(values) >= 30:
            momentum["1_month"] = float((values[-1] - values[-30]) / values[-30] * 100)
        if len(values) >= 90:
            momentum["3_months"] = float((values[-1] - values[-90]) / values[-90] * 100)
        if len(values) >= 252:
            momentum["1_year"] = float((values[-1] - values[-252]) / values[-252] * 100)
        
        result = {
            "commodity": commodity_name,
            "trend_direction": trend_direction,
            "trend_strength": float(r_squared),
            "slope": float(model.coef_[0]),
            "current_price": float(values[-1]),
            "ma_30": float(current_ma_short) if not pd.isna(current_ma_short) else None,
            "ma_90": float(current_ma_long) if not pd.isna(current_ma_long) else None,
            "momentum": momentum,
            "support_level": float(df['value'].rolling(window=window).min().iloc[-1]),
            "resistance_level": float(df['value'].rolling(window=window).max().iloc[-1])
        }
        
        return result
    
    def volatility_analysis(self, data: pd.DataFrame, commodity_name: str,
                          window: int = 30) -> Dict[str, Any]:
        """
        Calculate volatility metrics
        
        Args:
            data: DataFrame with 'value' column
            commodity_name: Name of the commodity
            window: Window for rolling calculations
        
        Returns:
            Dictionary with volatility metrics
        """
        if data.empty or len(data) < window:
            return {"error": f"Insufficient data for volatility analysis of {commodity_name}"}
        
        df = data.copy()
        
        # Calculate returns
        df['returns'] = df['value'].pct_change()
        df['log_returns'] = np.log(df['value'] / df['value'].shift(1))
        
        # Historical volatility (annualized)
        daily_vol = df['returns'].std()
        annual_vol = daily_vol * np.sqrt(252)  # 252 trading days
        
        # Rolling volatility
        df['rolling_vol'] = df['returns'].rolling(window=window).std() * np.sqrt(252)
        
        # Volatility regime
        current_vol = df['rolling_vol'].iloc[-1]
        avg_vol = df['rolling_vol'].mean()
        
        if current_vol > avg_vol * 1.5:
            vol_regime = "high"
        elif current_vol < avg_vol * 0.5:
            vol_regime = "low"
        else:
            vol_regime = "normal"
        
        result = {
            "commodity": commodity_name,
            "daily_volatility": float(daily_vol * 100),
            "annual_volatility": float(annual_vol * 100),
            "current_volatility": float(current_vol * 100) if not pd.isna(current_vol) else None,
            "average_volatility": float(avg_vol * 100),
            "volatility_regime": vol_regime,
            "max_daily_gain": float(df['returns'].max() * 100),
            "max_daily_loss": float(df['returns'].min() * 100)
        }
        
        return result
    
    def pattern_detection(self, data: pd.DataFrame, commodity_name: str) -> Dict[str, Any]:
        """
        Detect patterns including seasonality and cycles
        
        Args:
            data: DataFrame with 'value' column
            commodity_name: Name of the commodity
        
        Returns:
            Dictionary with pattern information
        """
        if data.empty or len(data) < 365:
            return {"error": f"Insufficient data for pattern detection of {commodity_name}"}
        
        df = data.copy()
        patterns = {"commodity": commodity_name}
        
        # Seasonal decomposition
        try:
            # Resample to daily if not already
            df_daily = df.resample('D').mean().interpolate()
            
            if len(df_daily) >= 730:  # Need at least 2 years
                decomposition = seasonal_decompose(
                    df_daily['value'], 
                    model='multiplicative', 
                    period=365,
                    extrapolate_trend='freq'
                )
                
                patterns['has_seasonality'] = True
                patterns['seasonal_strength'] = float(decomposition.seasonal.std())
                patterns['trend_strength'] = float(decomposition.trend.dropna().std())
                
                # Find peak and trough months
                seasonal_avg = decomposition.seasonal.groupby(decomposition.seasonal.index.month).mean()
                patterns['peak_month'] = int(seasonal_avg.idxmax())
                patterns['trough_month'] = int(seasonal_avg.idxmin())
            else:
                patterns['has_seasonality'] = False
                patterns['note'] = "Insufficient data for seasonal decomposition"
                
        except Exception as e:
            patterns['seasonality_error'] = str(e)
            patterns['has_seasonality'] = False
        
        # Price level analysis
        current_price = df['value'].iloc[-1]
        historical_mean = df['value'].mean()
        historical_std = df['value'].std()
        
        z_score = (current_price - historical_mean) / historical_std
        
        if z_score > 2:
            price_level = "extremely_high"
        elif z_score > 1:
            price_level = "high"
        elif z_score < -2:
            price_level = "extremely_low"
        elif z_score < -1:
            price_level = "low"
        else:
            price_level = "normal"
        
        patterns['price_level'] = price_level
        patterns['z_score'] = float(z_score)
        
        # Identify recent peaks and troughs
        df['peak'] = df['value'].rolling(window=30, center=True).max()
        df['trough'] = df['value'].rolling(window=30, center=True).min()
        
        patterns['recent_peak'] = float(df['peak'].iloc[-1])
        patterns['recent_trough'] = float(df['trough'].iloc[-1])
        
        return patterns
    
    def correlation_analysis(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze correlations between commodities
        
        Args:
            data_dict: Dictionary of commodity DataFrames
        
        Returns:
            Dictionary with correlation information
        """
        if len(data_dict) < 2:
            return {"error": "Need at least 2 commodities for correlation analysis"}
        
        # Create combined dataframe
        combined_df = pd.DataFrame()
        
        for commodity, df in data_dict.items():
            if not df.empty:
                combined_df[commodity] = df['value']
        
        # Align dates
        combined_df = combined_df.dropna()
        
        if combined_df.empty:
            return {"error": "No overlapping data for correlation analysis"}
        
        # Calculate correlation matrix
        corr_matrix = combined_df.corr()
        
        # Find strongest correlations
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlations.append({
                    "pair": f"{corr_matrix.columns[i]}_vs_{corr_matrix.columns[j]}",
                    "correlation": float(corr_matrix.iloc[i, j])
                })
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Calculate rolling correlations for top pairs
        rolling_corr = {}
        if len(correlations) > 0:
            top_pair = correlations[0]['pair'].split('_vs_')
            if len(top_pair) == 2:
                col1, col2 = top_pair
                if col1 in combined_df.columns and col2 in combined_df.columns:
                    rolling_corr['top_pair'] = correlations[0]['pair']
                    rolling_30d = combined_df[col1].rolling(30).corr(combined_df[col2])
                    rolling_corr['current_30d'] = float(rolling_30d.iloc[-1]) if not pd.isna(rolling_30d.iloc[-1]) else None
        
        result = {
            "correlation_matrix": corr_matrix.to_dict(),
            "top_correlations": correlations[:5],
            "rolling_correlations": rolling_corr
        }
        
        return result
    
    def analyze_all(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Perform all analyses on all commodities
        
        Args:
            data_dict: Dictionary of commodity DataFrames
        
        Returns:
            Comprehensive analysis results
        """
        results = {
            "statistical": {},
            "trends": {},
            "volatility": {},
            "patterns": {},
            "correlations": None
        }
        
        for commodity, data in data_dict.items():
            print(f"Analyzing {commodity}...")
            
            results["statistical"][commodity] = self.statistical_summary(data, commodity)
            results["trends"][commodity] = self.trend_analysis(data, commodity)
            results["volatility"][commodity] = self.volatility_analysis(data, commodity)
            results["patterns"][commodity] = self.pattern_detection(data, commodity)
        
        # Correlation analysis across all commodities
        results["correlations"] = self.correlation_analysis(data_dict)
        
        return results

# Test function
if __name__ == "__main__":
    print("Testing Commodity Analyzer...")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    sample_data = pd.DataFrame({
        'value': 1000 + np.cumsum(np.random.randn(len(dates)) * 10)
    }, index=dates)
    
    analyzer = CommodityAnalyzer()
    
    # Test statistical summary
    print("\n1. Statistical Summary:")
    stats = analyzer.statistical_summary(sample_data, "Test_Commodity")
    print(f"   Mean: ${stats['mean']:.2f}")
    print(f"   Std Dev: ${stats['std']:.2f}")
    print(f"   Range: ${stats['range']:.2f}")
    
    # Test trend analysis
    print("\n2. Trend Analysis:")
    trends = analyzer.trend_analysis(sample_data, "Test_Commodity")
    print(f"   Direction: {trends['trend_direction']}")
    print(f"   Strength: {trends['trend_strength']:.3f}")
    
    # Test volatility
    print("\n3. Volatility Analysis:")
    vol = analyzer.volatility_analysis(sample_data, "Test_Commodity")
    print(f"   Annual Volatility: {vol['annual_volatility']:.2f}%")
    print(f"   Regime: {vol['volatility_regime']}")
    
    print("\n" + "=" * 60)
    print("✓ Commodity Analyzer test complete")