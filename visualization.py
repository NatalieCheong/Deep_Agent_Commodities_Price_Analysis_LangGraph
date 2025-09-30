"""
Visualization Module
Creates charts and plots for commodity analysis
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

class Visualizer:
    """Creates visualizations for commodity analysis"""
    
    def __init__(self, output_dir: str = "./results/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.charts_created = []
    
    def plot_price_history(self, data_dict: Dict[str, pd.DataFrame],
                          title: str = "Commodity Price History") -> str:
        """Plot historical prices for all commodities"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, (commodity, data) in enumerate(data_dict.items()):
            if idx < len(axes) and not data.empty:
                ax = axes[idx]
                ax.plot(data.index, data['value'], linewidth=1.5, color='steelblue')
                ax.set_title(f"{commodity}", fontsize=12, fontweight='bold')
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                x_numeric = np.arange(len(data))
                z = np.polyfit(x_numeric, data['value'], 1)
                p = np.poly1d(z)
                ax.plot(data.index, p(x_numeric), "r--", alpha=0.5, linewidth=1, label='Trend')
                ax.legend()
        
        for idx in range(len(data_dict), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filename = self.output_dir / "price_history.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.charts_created.append(str(filename))
        return str(filename)
    
    def plot_returns_distribution(self, data_dict: Dict[str, pd.DataFrame],
                                 title: str = "Returns Distribution") -> str:
        """Plot returns distribution for all commodities"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, (commodity, data) in enumerate(data_dict.items()):
            if idx < len(axes) and not data.empty:
                ax = axes[idx]
                returns = data['value'].pct_change().dropna() * 100
                
                ax.hist(returns, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
                ax.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
                ax.axvline(returns.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {returns.median():.2f}%')
                
                ax.set_title(f"{commodity}", fontsize=12, fontweight='bold')
                ax.set_xlabel("Daily Returns (%)")
                ax.set_ylabel("Frequency")
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        for idx in range(len(data_dict), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filename = self.output_dir / "returns_distribution.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.charts_created.append(str(filename))
        return str(filename)
    
    def plot_correlation_matrix(self, data_dict: Dict[str, pd.DataFrame],
                               title: str = "Commodity Correlation Matrix") -> str:
        """Plot correlation matrix heatmap"""
        combined_df = pd.DataFrame()
        for commodity, data in data_dict.items():
            if not data.empty:
                combined_df[commodity] = data['value']
        
        corr_matrix = combined_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        filename = self.output_dir / "correlation_matrix.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.charts_created.append(str(filename))
        return str(filename)
    
    def plot_volatility(self, data_dict: Dict[str, pd.DataFrame],
                       window: int = 30,
                       title: str = "Rolling Volatility") -> str:
        """Plot rolling volatility for all commodities"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for commodity, data in data_dict.items():
            if not data.empty and len(data) > window:
                returns = data['value'].pct_change()
                rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
                ax.plot(data.index, rolling_vol, label=commodity, linewidth=1.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Annualized Volatility (%)")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = self.output_dir / "rolling_volatility.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.charts_created.append(str(filename))
        return str(filename)
    
    def plot_all_forecasts(self, data_dict: Dict[str, pd.DataFrame],
                          forecast_results: Dict[str, Any],
                          title: str = "Commodity Price Forecasts") -> str:
        """Plot all forecasts in one figure"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, (commodity, data) in enumerate(data_dict.items()):
            if idx < len(axes) and not data.empty and commodity in forecast_results:
                ax = axes[idx]
                forecast = forecast_results[commodity]
                
                ax.plot(data.index, data['value'], label='Historical', linewidth=1.5, color='steelblue')
                
                if 'error' not in forecast:
                    forecast_dates = pd.to_datetime(forecast['forecast']['dates'])
                    forecast_values = forecast['forecast']['values']
                    ax.plot(forecast_dates, forecast_values, label='Forecast', 
                           linewidth=1.5, color='orange', linestyle='--')
                    
                    if 'lower_bound' in forecast['forecast']:
                        lower = forecast['forecast']['lower_bound']
                        upper = forecast['forecast']['upper_bound']
                        ax.fill_between(forecast_dates, lower, upper, alpha=0.2, color='orange')
                
                ax.set_title(f"{commodity}", fontsize=12, fontweight='bold')
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
        
        for idx in range(len(data_dict), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filename = self.output_dir / "all_forecasts.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.charts_created.append(str(filename))
        return str(filename)
    
    def plot_risk_metrics(self, risk_results: Dict[str, Any],
                         title: str = "Risk Metrics Comparison") -> str:
        """Plot risk metrics comparison"""
        if 'individual' not in risk_results:
            return ""
        
        commodities = []
        var_values = []
        sharpe_values = []
        max_dd_values = []
        
        for commodity, metrics in risk_results['individual'].items():
            if 'error' not in metrics.get('var', {}) and 'error' not in metrics.get('sharpe', {}):
                commodities.append(commodity)
                var_values.append(abs(metrics['var']['var_historical']['percentage']))
                sharpe_values.append(metrics['sharpe']['sharpe_ratio'])
                max_dd_values.append(abs(metrics['drawdown']['max_drawdown']['percentage']))
        
        if not commodities:
            return ""
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        axes[0].barh(commodities, var_values, color='coral')
        axes[0].set_xlabel("VaR (95%) %")
        axes[0].set_title("Value at Risk Comparison", fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        colors = ['green' if s > 1 else 'orange' if s > 0 else 'red' for s in sharpe_values]
        axes[1].barh(commodities, sharpe_values, color=colors)
        axes[1].set_xlabel("Sharpe Ratio")
        axes[1].set_title("Risk-Adjusted Returns", fontweight='bold')
        axes[1].axvline(x=1, color='gray', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        axes[2].barh(commodities, max_dd_values, color='steelblue')
        axes[2].set_xlabel("Max Drawdown %")
        axes[2].set_title("Maximum Drawdown", fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = self.output_dir / "risk_metrics_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.charts_created.append(str(filename))
        return str(filename)
    
    def create_all_visualizations(self, data_dict: Dict[str, pd.DataFrame],
                                 forecast_results: Dict[str, Any] = None,
                                 risk_results: Dict[str, Any] = None) -> List[str]:
        """Create all standard visualizations"""
        print("\nCreating visualizations...")
        
        charts = []
        
        print("  - Price history chart...")
        charts.append(self.plot_price_history(data_dict))
        
        print("  - Returns distribution...")
        charts.append(self.plot_returns_distribution(data_dict))
        
        print("  - Correlation matrix...")
        charts.append(self.plot_correlation_matrix(data_dict))
        
        print("  - Volatility analysis...")
        charts.append(self.plot_volatility(data_dict))
        
        if forecast_results:
            print("  - Forecast charts...")
            charts.append(self.plot_all_forecasts(data_dict, forecast_results))
        
        if risk_results:
            print("  - Risk metrics...")
            charts.append(self.plot_risk_metrics(risk_results))
        
        print(f"  ✓ Created {len(charts)} charts")
        
        return charts

if __name__ == "__main__":
    print("Testing Visualizer...")
    print("=" * 60)
    
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    test_data = {
        'Gold': pd.DataFrame({'value': 1800 + np.cumsum(np.random.randn(len(dates)) * 10)}, index=dates),
        'Crude_Oil': pd.DataFrame({'value': 70 + np.cumsum(np.random.randn(len(dates)) * 2)}, index=dates)
    }
    
    viz = Visualizer("./test_charts")
    
    print("\nCreating test charts...")
    viz.plot_price_history(test_data)
    viz.plot_correlation_matrix(test_data)
    
    print(f"\n✓ Created {len(viz.charts_created)} test charts")
    print("=" * 60)