"""
Forecasting Module
Price forecasting using Prophet and statistical models
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import Config
import warnings
warnings.filterwarnings('ignore')

class CommodityForecaster:
    """Forecasts commodity prices using multiple methods"""
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
    
    def forecast_prophet(self, data: pd.DataFrame, commodity_name: str,
                        periods: int = None) -> Dict[str, Any]:
        """
        Forecast using Facebook Prophet
        
        Args:
            data: DataFrame with date index and 'value' column
            commodity_name: Name of the commodity
            periods: Number of days to forecast (uses Config if None)
        
        Returns:
            Dictionary with forecast results
        """
        if data.empty or len(data) < 730:  # Need at least 2 years
            return {"error": f"Insufficient data for forecasting {commodity_name}"}
        
        periods = periods or Config.FORECAST_DAYS
        
        # Prepare data for Prophet
        df_prophet = data.reset_index()
        df_prophet.columns = ['ds', 'y']
        
        try:
            # Initialize and fit Prophet model
            model = Prophet(
                changepoint_prior_scale=Config.PROPHET_CHANGEPOINT_PRIOR,
                seasonality_prior_scale=Config.PROPHET_SEASONALITY_PRIOR,
                holidays_prior_scale=Config.PROPHET_HOLIDAYS_PRIOR,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            
            model.fit(df_prophet)
            
            # Make future dataframe
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            # Extract forecast values
            forecast_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            
            # Calculate confidence intervals
            confidence_width = (forecast_values['yhat_upper'] - forecast_values['yhat_lower']).mean()
            avg_price = forecast_values['yhat'].mean()
            confidence_score = max(0, min(100, 100 - (confidence_width / avg_price * 100)))
            
            # Store model
            self.models[commodity_name] = model
            
            result = {
                "commodity": commodity_name,
                "method": "Prophet",
                "forecast_periods": periods,
                "forecast": {
                    "dates": forecast_values['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    "values": forecast_values['yhat'].tolist(),
                    "lower_bound": forecast_values['yhat_lower'].tolist(),
                    "upper_bound": forecast_values['yhat_upper'].tolist()
                },
                "summary": {
                    "mean_forecast": float(forecast_values['yhat'].mean()),
                    "min_forecast": float(forecast_values['yhat'].min()),
                    "max_forecast": float(forecast_values['yhat'].max()),
                    "final_forecast": float(forecast_values['yhat'].iloc[-1]),
                    "confidence_score": float(confidence_score)
                },
                "current_price": float(data['value'].iloc[-1]),
                "expected_change": float(
                    (forecast_values['yhat'].iloc[-1] - data['value'].iloc[-1]) / 
                    data['value'].iloc[-1] * 100
                )
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Prophet forecasting failed for {commodity_name}: {str(e)}"}
    
    def forecast_linear(self, data: pd.DataFrame, commodity_name: str,
                       periods: int = None) -> Dict[str, Any]:
        """
        Simple linear trend forecast as baseline
        
        Args:
            data: DataFrame with date index and 'value' column
            commodity_name: Name of the commodity
            periods: Number of days to forecast
        
        Returns:
            Dictionary with forecast results
        """
        if data.empty or len(data) < 30:
            return {"error": f"Insufficient data for linear forecast of {commodity_name}"}
        
        periods = periods or Config.FORECAST_DAYS
        
        # Fit linear model to recent data (last year)
        recent_data = data.tail(365) if len(data) > 365 else data
        X = np.arange(len(recent_data)).reshape(-1, 1)
        y = recent_data['value'].values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast
        future_X = np.arange(len(recent_data), len(recent_data) + periods).reshape(-1, 1)
        forecast_values = model.predict(future_X)
        
        # Generate future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        
        # Estimate confidence interval (simple approach)
        residuals = y - model.predict(X)
        std_error = np.std(residuals)
        
        result = {
            "commodity": commodity_name,
            "method": "Linear",
            "forecast_periods": periods,
            "forecast": {
                "dates": future_dates.strftime('%Y-%m-%d').tolist(),
                "values": forecast_values.tolist(),
                "std_error": float(std_error)
            },
            "summary": {
                "mean_forecast": float(np.mean(forecast_values)),
                "final_forecast": float(forecast_values[-1]),
                "slope": float(model.coef_[0])
            },
            "current_price": float(data['value'].iloc[-1]),
            "expected_change": float(
                (forecast_values[-1] - data['value'].iloc[-1]) / 
                data['value'].iloc[-1] * 100
            )
        }
        
        return result
    
    def forecast_moving_average(self, data: pd.DataFrame, commodity_name: str,
                               window: int = 90, periods: int = None) -> Dict[str, Any]:
        """
        Forecast using moving average extrapolation
        
        Args:
            data: DataFrame with date index and 'value' column
            commodity_name: Name of the commodity
            window: Moving average window
            periods: Number of days to forecast
        
        Returns:
            Dictionary with forecast results
        """
        if data.empty or len(data) < window:
            return {"error": f"Insufficient data for MA forecast of {commodity_name}"}
        
        periods = periods or min(365, Config.FORECAST_DAYS)  # Limit MA forecast to 1 year
        
        # Calculate moving average
        ma = data['value'].rolling(window=window).mean()
        current_ma = ma.iloc[-1]
        
        # Simple extrapolation: assume MA continues
        forecast_values = np.full(periods, current_ma)
        
        # Add some trend if detected
        recent_ma = ma.tail(30)
        if len(recent_ma) > 1:
            ma_trend = (recent_ma.iloc[-1] - recent_ma.iloc[0]) / len(recent_ma)
            trend_component = np.arange(1, periods + 1) * ma_trend
            forecast_values = forecast_values + trend_component
        
        # Generate future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        
        result = {
            "commodity": commodity_name,
            "method": f"Moving_Average_{window}d",
            "forecast_periods": periods,
            "forecast": {
                "dates": future_dates.strftime('%Y-%m-%d').tolist(),
                "values": forecast_values.tolist()
            },
            "summary": {
                "mean_forecast": float(np.mean(forecast_values)),
                "final_forecast": float(forecast_values[-1])
            },
            "current_price": float(data['value'].iloc[-1]),
            "expected_change": float(
                (forecast_values[-1] - data['value'].iloc[-1]) / 
                data['value'].iloc[-1] * 100
            )
        }
        
        return result
    
    def ensemble_forecast(self, data: pd.DataFrame, commodity_name: str,
                         periods: int = None) -> Dict[str, Any]:
        """
        Ensemble forecast combining multiple methods
        
        Args:
            data: DataFrame with date index and 'value' column
            commodity_name: Name of the commodity
            periods: Number of days to forecast
        
        Returns:
            Dictionary with ensemble forecast results
        """
        periods = periods or Config.FORECAST_DAYS
        
        # Get forecasts from different methods
        prophet_result = self.forecast_prophet(data, commodity_name, periods)
        linear_result = self.forecast_linear(data, commodity_name, periods)
        ma_result = self.forecast_moving_average(data, commodity_name, 90, min(365, periods))
        
        # Check if any method failed
        if "error" in prophet_result:
            return prophet_result  # Return error if Prophet fails
        
        # Combine forecasts (weighted average)
        weights = {"prophet": 0.6, "linear": 0.2, "ma": 0.2}
        
        prophet_values = np.array(prophet_result["forecast"]["values"])
        linear_values = np.array(linear_result["forecast"]["values"])
        
        # MA might have fewer periods
        ma_periods = len(ma_result["forecast"]["values"])
        ensemble_values = np.zeros(periods)
        
        for i in range(periods):
            if i < ma_periods:
                ensemble_values[i] = (
                    weights["prophet"] * prophet_values[i] +
                    weights["linear"] * linear_values[i] +
                    weights["ma"] * ma_result["forecast"]["values"][i]
                )
            else:
                # Only Prophet and Linear for longer periods
                total_weight = weights["prophet"] + weights["linear"]
                ensemble_values[i] = (
                    weights["prophet"] / total_weight * prophet_values[i] +
                    weights["linear"] / total_weight * linear_values[i]
                )
        
        future_dates = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1), 
            periods=periods
        )
        
        result = {
            "commodity": commodity_name,
            "method": "Ensemble",
            "forecast_periods": periods,
            "forecast": {
                "dates": future_dates.strftime('%Y-%m-%d').tolist(),
                "values": ensemble_values.tolist(),
                "lower_bound": prophet_result["forecast"]["lower_bound"],
                "upper_bound": prophet_result["forecast"]["upper_bound"]
            },
            "component_forecasts": {
                "prophet": prophet_result["summary"]["final_forecast"],
                "linear": linear_result["summary"]["final_forecast"],
                "ma": ma_result["summary"]["final_forecast"]
            },
            "summary": {
                "mean_forecast": float(np.mean(ensemble_values)),
                "min_forecast": float(np.min(ensemble_values)),
                "max_forecast": float(np.max(ensemble_values)),
                "final_forecast": float(ensemble_values[-1])
            },
            "current_price": float(data['value'].iloc[-1]),
            "expected_change": float(
                (ensemble_values[-1] - data['value'].iloc[-1]) / 
                data['value'].iloc[-1] * 100
            )
        }
        
        return result
    
    def forecast_all(self, data_dict: Dict[str, pd.DataFrame],
                    method: str = "ensemble") -> Dict[str, Any]:
        """
        Forecast all commodities
        
        Args:
            data_dict: Dictionary of commodity DataFrames
            method: Forecasting method ('prophet', 'ensemble', etc.)
        
        Returns:
            Dictionary with all forecasts
        """
        results = {}
        
        for commodity, data in data_dict.items():
            print(f"Forecasting {commodity}...")
            
            if method == "ensemble":
                results[commodity] = self.ensemble_forecast(data, commodity)
            elif method == "prophet":
                results[commodity] = self.forecast_prophet(data, commodity)
            elif method == "linear":
                results[commodity] = self.forecast_linear(data, commodity)
            else:
                results[commodity] = self.ensemble_forecast(data, commodity)
        
        return results

# Test function
if __name__ == "__main__":
    print("Testing Commodity Forecaster...")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    sample_data = pd.DataFrame({
        'value': 1000 + np.cumsum(np.random.randn(len(dates)) * 10)
    }, index=dates)
    
    forecaster = CommodityForecaster()
    
    # Test Prophet forecast
    print("\n1. Prophet Forecast:")
    result = forecaster.forecast_prophet(sample_data, "Test_Commodity", periods=90)
    if "error" not in result:
        print(f"   Mean forecast: ${result['summary']['mean_forecast']:.2f}")
        print(f"   Final forecast: ${result['summary']['final_forecast']:.2f}")
        print(f"   Expected change: {result['expected_change']:.2f}%")
    
    print("\n" + "=" * 60)
    print("✓ Forecaster test complete")