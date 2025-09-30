"""
Risk Analysis Module
Value at Risk (VaR), CVaR, and stress testing
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy import stats
from config import Config
import warnings
warnings.filterwarnings('ignore')

class RiskAnalyzer:
    """Performs risk analysis on commodity portfolios"""
    
    def __init__(self):
        self.risk_metrics = {}
    
    def calculate_var(self, data: pd.DataFrame, commodity_name: str,
                     confidence_level: float = None) -> Dict[str, Any]:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            data: DataFrame with 'value' column
            commodity_name: Name of the commodity
            confidence_level: Confidence level (0.95 = 95%)
        
        Returns:
            Dictionary with VaR metrics
        """
        if data.empty or len(data) < 30:
            return {"error": f"Insufficient data for VaR calculation of {commodity_name}"}
        
        confidence_level = confidence_level or Config.VAR_CONFIDENCE_LEVEL
        
        # Calculate returns
        returns = data['value'].pct_change().dropna()
        
        # Historical VaR (non-parametric)
        var_historical = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Parametric VaR (assumes normal distribution)
        mean_return = returns.mean()
        std_return = returns.std()
        var_parametric = stats.norm.ppf(1 - confidence_level, mean_return, std_return)
        
        # VaR in dollar terms (based on current price)
        current_price = data['value'].iloc[-1]
        var_dollar_historical = current_price * var_historical
        var_dollar_parametric = current_price * var_parametric
        
        result = {
            "commodity": commodity_name,
            "confidence_level": confidence_level,
            "current_price": float(current_price),
            "var_historical": {
                "percentage": float(var_historical * 100),
                "dollar_amount": float(var_dollar_historical)
            },
            "var_parametric": {
                "percentage": float(var_parametric * 100),
                "dollar_amount": float(var_dollar_parametric)
            },
            "interpretation": f"With {confidence_level*100:.0f}% confidence, " +
                            f"daily loss will not exceed {abs(var_historical)*100:.2f}%"
        }
        
        return result
    
    def calculate_cvar(self, data: pd.DataFrame, commodity_name: str,
                      confidence_level: float = None) -> Dict[str, Any]:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall)
        
        Args:
            data: DataFrame with 'value' column
            commodity_name: Name of the commodity
            confidence_level: Confidence level (0.95 = 95%)
        
        Returns:
            Dictionary with CVaR metrics
        """
        if data.empty or len(data) < 30:
            return {"error": f"Insufficient data for CVaR calculation of {commodity_name}"}
        
        confidence_level = confidence_level or Config.CVAR_CONFIDENCE_LEVEL
        
        # Calculate returns
        returns = data['value'].pct_change().dropna()
        
        # Calculate VaR first
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        
        # CVaR is the average of returns below VaR threshold
        tail_returns = returns[returns <= var_threshold]
        cvar = tail_returns.mean()
        
        # CVaR in dollar terms
        current_price = data['value'].iloc[-1]
        cvar_dollar = current_price * cvar
        
        result = {
            "commodity": commodity_name,
            "confidence_level": confidence_level,
            "current_price": float(current_price),
            "cvar": {
                "percentage": float(cvar * 100),
                "dollar_amount": float(cvar_dollar)
            },
            "var_for_reference": float(var_threshold * 100),
            "interpretation": f"If loss exceeds VaR, expected loss is {abs(cvar)*100:.2f}%"
        }
        
        return result
    
    def calculate_drawdown(self, data: pd.DataFrame, commodity_name: str) -> Dict[str, Any]:
        """
        Calculate maximum drawdown and current drawdown
        
        Args:
            data: DataFrame with 'value' column
            commodity_name: Name of the commodity
        
        Returns:
            Dictionary with drawdown metrics
        """
        if data.empty:
            return {"error": f"No data for drawdown calculation of {commodity_name}"}
        
        # Calculate cumulative maximum
        cummax = data['value'].cummax()
        
        # Calculate drawdown
        drawdown = (data['value'] - cummax) / cummax
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        max_drawdown_date = drawdown.idxmin()
        
        # Current drawdown
        current_drawdown = drawdown.iloc[-1]
        
        # Recovery analysis
        is_recovering = current_drawdown > drawdown.iloc[-30:].min()
        
        # Time in drawdown
        days_in_drawdown = 0
        if current_drawdown < 0:
            for i in range(len(data)-1, -1, -1):
                if drawdown.iloc[i] >= 0:
                    break
                days_in_drawdown += 1
        
        result = {
            "commodity": commodity_name,
            "current_price": float(data['value'].iloc[-1]),
            "peak_price": float(cummax.iloc[-1]),
            "max_drawdown": {
                "percentage": float(max_drawdown * 100),
                "date": max_drawdown_date.strftime("%Y-%m-%d")
            },
            "current_drawdown": {
                "percentage": float(current_drawdown * 100),
                "days_in_drawdown": days_in_drawdown,
                "is_recovering": is_recovering
            }
        }
        
        return result
    
    def stress_test(self, data: pd.DataFrame, commodity_name: str,
                   scenarios: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Perform stress testing with various scenarios
        
        Args:
            data: DataFrame with 'value' column
            commodity_name: Name of the commodity
            scenarios: Dictionary of scenario names and price shocks (e.g., {"recession": -0.30})
        
        Returns:
            Dictionary with stress test results
        """
        if data.empty:
            return {"error": f"No data for stress testing of {commodity_name}"}
        
        current_price = data['value'].iloc[-1]
        
        # Default scenarios if none provided
        if scenarios is None:
            scenarios = {
                "mild_decline": -0.10,
                "moderate_decline": -0.20,
                "severe_decline": -0.30,
                "market_crash": -0.50,
                "mild_increase": 0.10,
                "moderate_increase": 0.20,
                "strong_rally": 0.30
            }
        
        results = {
            "commodity": commodity_name,
            "current_price": float(current_price),
            "scenarios": {}
        }
        
        for scenario_name, shock in scenarios.items():
            stressed_price = current_price * (1 + shock)
            
            results["scenarios"][scenario_name] = {
                "shock_percentage": float(shock * 100),
                "stressed_price": float(stressed_price),
                "price_change": float(stressed_price - current_price),
                "percentage_change": float(shock * 100)
            }
        
        return results
    
    def calculate_sharpe_ratio(self, data: pd.DataFrame, commodity_name: str,
                              risk_free_rate: float = 0.03) -> Dict[str, Any]:
        """
        Calculate Sharpe Ratio
        
        Args:
            data: DataFrame with 'value' column
            commodity_name: Name of the commodity
            risk_free_rate: Annual risk-free rate (default 3%)
        
        Returns:
            Dictionary with Sharpe ratio
        """
        if data.empty or len(data) < 30:
            return {"error": f"Insufficient data for Sharpe ratio of {commodity_name}"}
        
        # Calculate returns
        returns = data['value'].pct_change().dropna()
        
        # Annualized metrics
        annual_return = returns.mean() * 252  # 252 trading days
        annual_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        result = {
            "commodity": commodity_name,
            "annual_return": float(annual_return * 100),
            "annual_volatility": float(annual_volatility * 100),
            "risk_free_rate": float(risk_free_rate * 100),
            "sharpe_ratio": float(sharpe_ratio),
            "interpretation": self._interpret_sharpe(sharpe_ratio)
        }
        
        return result
    
    def _interpret_sharpe(self, sharpe: float) -> str:
        """Interpret Sharpe ratio value"""
        if sharpe > 2:
            return "Excellent risk-adjusted returns"
        elif sharpe > 1:
            return "Good risk-adjusted returns"
        elif sharpe > 0:
            return "Acceptable risk-adjusted returns"
        else:
            return "Poor risk-adjusted returns"
    
    def portfolio_risk(self, data_dict: Dict[str, pd.DataFrame],
                      weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Calculate portfolio-level risk metrics
        
        Args:
            data_dict: Dictionary of commodity DataFrames
            weights: Portfolio weights (equal weight if None)
        
        Returns:
            Dictionary with portfolio risk metrics
        """
        if len(data_dict) < 2:
            return {"error": "Need at least 2 commodities for portfolio analysis"}
        
        # Default to equal weights
        if weights is None:
            n = len(data_dict)
            weights = {commodity: 1.0/n for commodity in data_dict.keys()}
        
        # Create returns dataframe
        returns_df = pd.DataFrame()
        for commodity, data in data_dict.items():
            if not data.empty:
                returns_df[commodity] = data['value'].pct_change()
        
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            return {"error": "No overlapping data for portfolio analysis"}
        
        # Calculate portfolio returns
        weight_array = np.array([weights.get(col, 0) for col in returns_df.columns])
        portfolio_returns = returns_df.values @ weight_array
        
        # Portfolio metrics
        portfolio_mean = portfolio_returns.mean() * 252
        portfolio_std = portfolio_returns.std() * np.sqrt(252)
        portfolio_sharpe = portfolio_mean / portfolio_std if portfolio_std > 0 else 0
        
        # Portfolio VaR
        portfolio_var_95 = np.percentile(portfolio_returns, 5)
        
        # Correlation matrix
        corr_matrix = returns_df.corr()
        avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        result = {
            "portfolio_composition": weights,
            "annual_return": float(portfolio_mean * 100),
            "annual_volatility": float(portfolio_std * 100),
            "sharpe_ratio": float(portfolio_sharpe),
            "var_95": float(portfolio_var_95 * 100),
            "average_correlation": float(avg_correlation),
            "diversification_benefit": "High" if avg_correlation < 0.5 else "Moderate" if avg_correlation < 0.7 else "Low"
        }
        
        return result
    
    def analyze_all_risks(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis on all commodities
        
        Args:
            data_dict: Dictionary of commodity DataFrames
        
        Returns:
            Dictionary with all risk metrics
        """
        results = {
            "individual": {},
            "portfolio": None
        }
        
        for commodity, data in data_dict.items():
            print(f"Analyzing risk for {commodity}...")
            
            results["individual"][commodity] = {
                "var": self.calculate_var(data, commodity),
                "cvar": self.calculate_cvar(data, commodity),
                "drawdown": self.calculate_drawdown(data, commodity),
                "sharpe": self.calculate_sharpe_ratio(data, commodity),
                "stress_test": self.stress_test(data, commodity)
            }
        
        # Portfolio risk
        results["portfolio"] = self.portfolio_risk(data_dict)
        
        return results

# Test function
if __name__ == "__main__":
    print("Testing Risk Analyzer...")
    print("=" * 60)
    
    # Create sample data with some volatility
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    sample_data = pd.DataFrame({
        'value': 1000 + np.cumsum(np.random.randn(len(dates)) * 20)
    }, index=dates)
    
    analyzer = RiskAnalyzer()
    
    # Test VaR
    print("\n1. Value at Risk (VaR):")
    var = analyzer.calculate_var(sample_data, "Test_Commodity")
    print(f"   Historical VaR (95%): {var['var_historical']['percentage']:.2f}%")
    print(f"   {var['interpretation']}")
    
    # Test CVaR
    print("\n2. Conditional VaR (CVaR):")
    cvar = analyzer.calculate_cvar(sample_data, "Test_Commodity")
    print(f"   CVaR (95%): {cvar['cvar']['percentage']:.2f}%")
    
    # Test Drawdown
    print("\n3. Drawdown Analysis:")
    dd = analyzer.calculate_drawdown(sample_data, "Test_Commodity")
    print(f"   Max Drawdown: {dd['max_drawdown']['percentage']:.2f}%")
    print(f"   Current Drawdown: {dd['current_drawdown']['percentage']:.2f}%")
    
    # Test Sharpe Ratio
    print("\n4. Sharpe Ratio:")
    sharpe = analyzer.calculate_sharpe_ratio(sample_data, "Test_Commodity")
    print(f"   Sharpe Ratio: {sharpe['sharpe_ratio']:.2f}")
    print(f"   {sharpe['interpretation']}")
    
    print("\n" + "=" * 60)
    print("✓ Risk Analyzer test complete")