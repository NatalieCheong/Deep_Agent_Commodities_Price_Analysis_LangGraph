"""
Report Generator Module
Creates comprehensive analysis reports
"""
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
import json

class ReportGenerator:
    """Generates comprehensive analysis reports"""
    
    def __init__(self, output_dir: str = "./results/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_executive_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate executive summary section"""
        summary = []
        summary.append("# EXECUTIVE SUMMARY")
        summary.append("=" * 80)
        summary.append("")
        
        if 'statistical' in analysis_results:
            summary.append("## Key Findings")
            summary.append("")
            
            for commodity, stats in analysis_results['statistical'].items():
                if 'error' not in stats:
                    summary.append(f"### {commodity}")
                    summary.append(f"- Current Price: ${stats.get('mean', 0):.2f}")
                    if 'recent_90d' in stats:
                        summary.append(f"- 90-Day Change: {stats['recent_90d'].get('change_pct', 0):.2f}%")
                    summary.append(f"- Volatility (CV): {stats.get('cv', 0):.2f}%")
                    summary.append("")
        
        return "\n".join(summary)
    
    def generate_data_quality_section(self, data_quality: Dict[str, Any]) -> str:
        """Generate data quality section"""
        section = []
        section.append("# DATA QUALITY REPORT")
        section.append("=" * 80)
        section.append("")
        
        for commodity, metrics in data_quality.items():
            if metrics.get('status') == 'ok':
                section.append(f"## {commodity}")
                section.append(f"- Records: {metrics['records']}")
                section.append(f"- Date Range: {metrics['date_range']['start']} to {metrics['date_range']['end']}")
                section.append(f"- Completeness: {metrics['completeness']:.1f}%")
                section.append(f"- Latest Price: ${metrics['latest_price']:.2f} ({metrics['latest_date']})")
                section.append("")
        
        return "\n".join(section)
    
    def generate_statistical_analysis(self, stats_results: Dict[str, Any]) -> str:
        """Generate statistical analysis section"""
        section = []
        section.append("# STATISTICAL ANALYSIS")
        section.append("=" * 80)
        section.append("")
        
        for commodity, stats in stats_results.items():
            if 'error' not in stats:
                section.append(f"## {commodity}")
                section.append(f"- Mean: ${stats['mean']:.2f}")
                section.append(f"- Median: ${stats['median']:.2f}")
                section.append(f"- Std Dev: ${stats['std']:.2f}")
                section.append(f"- Range: ${stats['min']:.2f} - ${stats['max']:.2f}")
                section.append(f"- Coefficient of Variation: {stats['cv']:.2f}%")
                section.append(f"- Skewness: {stats['skewness']:.3f}")
                section.append(f"- Kurtosis: {stats['kurtosis']:.3f}")
                section.append("")
        
        return "\n".join(section)
    
    def generate_trend_analysis(self, trend_results: Dict[str, Any]) -> str:
        """Generate trend analysis section"""
        section = []
        section.append("# TREND ANALYSIS")
        section.append("=" * 80)
        section.append("")
        
        for commodity, trend in trend_results.items():
            if 'error' not in trend:
                section.append(f"## {commodity}")
                section.append(f"- Trend Direction: {trend['trend_direction'].upper()}")
                section.append(f"- Trend Strength (R²): {trend['trend_strength']:.3f}")
                section.append(f"- Current Price: ${trend['current_price']:.2f}")
                
                if trend.get('ma_30'):
                    section.append(f"- 30-Day MA: ${trend['ma_30']:.2f}")
                if trend.get('ma_90'):
                    section.append(f"- 90-Day MA: ${trend['ma_90']:.2f}")
                
                if 'momentum' in trend:
                    section.append("- Momentum:")
                    for period, value in trend['momentum'].items():
                        section.append(f"  * {period}: {value:.2f}%")
                
                section.append("")
        
        return "\n".join(section)
    
    def generate_forecast_section(self, forecast_results: Dict[str, Any]) -> str:
        """Generate forecast section"""
        section = []
        section.append("# PRICE FORECASTS")
        section.append("=" * 80)
        section.append("")
        
        for commodity, forecast in forecast_results.items():
            if 'error' not in forecast:
                section.append(f"## {commodity}")
                section.append(f"- Method: {forecast.get('method', 'N/A')}")
                section.append(f"- Current Price: ${forecast['current_price']:.2f}")
                section.append(f"- Forecast Horizon: {forecast['forecast_periods']} days")
                
                if 'summary' in forecast:
                    summary = forecast['summary']
                    section.append(f"- Mean Forecast: ${summary.get('mean_forecast', 0):.2f}")
                    section.append(f"- Final Forecast: ${summary.get('final_forecast', 0):.2f}")
                    section.append(f"- Expected Change: {forecast.get('expected_change', 0):.2f}%")
                
                section.append("")
        
        return "\n".join(section)
    
    def generate_risk_assessment(self, risk_results: Dict[str, Any]) -> str:
        """Generate risk assessment section"""
        section = []
        section.append("# RISK ASSESSMENT")
        section.append("=" * 80)
        section.append("")
        
        if 'individual' in risk_results:
            for commodity, metrics in risk_results['individual'].items():
                section.append(f"## {commodity}")
                
                # VaR
                if 'var' in metrics and 'error' not in metrics['var']:
                    var = metrics['var']
                    section.append(f"### Value at Risk (VaR)")
                    section.append(f"- Historical VaR (95%): {var['var_historical']['percentage']:.2f}%")
                    section.append(f"- Dollar VaR: ${abs(var['var_historical']['dollar_amount']):.2f}")
                
                # Sharpe Ratio
                if 'sharpe' in metrics and 'error' not in metrics['sharpe']:
                    sharpe = metrics['sharpe']
                    section.append(f"### Risk-Adjusted Returns")
                    section.append(f"- Sharpe Ratio: {sharpe['sharpe_ratio']:.3f}")
                    section.append(f"- Interpretation: {sharpe['interpretation']}")
                
                # Drawdown
                if 'drawdown' in metrics and 'error' not in metrics['drawdown']:
                    dd = metrics['drawdown']
                    section.append(f"### Drawdown Analysis")
                    section.append(f"- Max Drawdown: {dd['max_drawdown']['percentage']:.2f}%")
                    section.append(f"- Current Drawdown: {dd['current_drawdown']['percentage']:.2f}%")
                
                section.append("")
        
        # Portfolio risk
        if 'portfolio' in risk_results and risk_results['portfolio']:
            portfolio = risk_results['portfolio']
            if 'error' not in portfolio:
                section.append("## Portfolio Risk")
                section.append(f"- Annual Return: {portfolio.get('annual_return', 0):.2f}%")
                section.append(f"- Annual Volatility: {portfolio.get('annual_volatility', 0):.2f}%")
                section.append(f"- Sharpe Ratio: {portfolio.get('sharpe_ratio', 0):.3f}")
                section.append(f"- Portfolio VaR (95%): {portfolio.get('var_95', 0):.2f}%")
                section.append(f"- Diversification: {portfolio.get('diversification_benefit', 'N/A')}")
                section.append("")
        
        return "\n".join(section)
    
    def generate_recommendations(self, analysis_results: Dict[str, Any]) -> str:
        """Generate recommendations section"""
        section = []
        section.append("# RECOMMENDATIONS")
        section.append("=" * 80)
        section.append("")
        
        recommendations = []
        
        # Analyze trends and generate recommendations
        if 'trends' in analysis_results:
            for commodity, trend in analysis_results['trends'].items():
                if 'error' not in trend:
                    direction = trend.get('trend_direction', 'neutral')
                    strength = trend.get('trend_strength', 0)
                    
                    if direction == 'upward' and strength > 0.7:
                        recommendations.append(f"- {commodity}: Strong upward trend detected. Consider long positions.")
                    elif direction == 'downward' and strength > 0.7:
                        recommendations.append(f"- {commodity}: Strong downward trend detected. Exercise caution.")
                    elif strength < 0.3:
                        recommendations.append(f"- {commodity}: Weak trend. Market may be ranging or transitioning.")
        
        # Risk-based recommendations
        if 'individual' in analysis_results.get('risk', {}):
            for commodity, metrics in analysis_results['risk']['individual'].items():
                if 'sharpe' in metrics and 'error' not in metrics['sharpe']:
                    sharpe = metrics['sharpe']['sharpe_ratio']
                    if sharpe < 0:
                        recommendations.append(f"- {commodity}: Negative risk-adjusted returns. Review exposure.")
        
        if recommendations:
            section.extend(recommendations)
        else:
            section.append("- Continue monitoring market conditions")
            section.append("- Maintain diversified portfolio")
            section.append("- Review positions based on risk tolerance")
        
        section.append("")
        return "\n".join(section)
    
    def generate_full_report(self, 
                           data_quality: Dict[str, Any],
                           analysis_results: Dict[str, Any],
                           forecast_results: Dict[str, Any],
                           risk_results: Dict[str, Any],
                           charts: List[str] = None) -> str:
        """Generate complete analysis report"""
        
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("COMMODITY PRICE ANALYSIS REPORT")
        report.append("Deep Learning Agent - Comprehensive Analysis")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append(self.generate_executive_summary(analysis_results))
        report.append("")
        
        # Data Quality
        report.append(self.generate_data_quality_section(data_quality))
        report.append("")
        
        # Statistical Analysis
        if 'statistical' in analysis_results:
            report.append(self.generate_statistical_analysis(analysis_results['statistical']))
            report.append("")
        
        # Trend Analysis
        if 'trends' in analysis_results:
            report.append(self.generate_trend_analysis(analysis_results['trends']))
            report.append("")
        
        # Forecasts
        if forecast_results:
            report.append(self.generate_forecast_section(forecast_results))
            report.append("")
        
        # Risk Assessment
        if risk_results:
            report.append(self.generate_risk_assessment(risk_results))
            report.append("")
        
        # Recommendations
        combined_results = {**analysis_results, 'risk': risk_results}
        report.append(self.generate_recommendations(combined_results))
        report.append("")
        
        # Charts section
        if charts:
            report.append("# VISUALIZATIONS")
            report.append("=" * 80)
            report.append("")
            report.append("The following charts have been generated:")
            for chart in charts:
                report.append(f"- {Path(chart).name}")
            report.append("")
        
        # Footer
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f"commodity_analysis_report_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Report saved to: {filename}")
        
        return report_text
    
    def save_json_results(self, all_results: Dict[str, Any]) -> str:
        """Save all results as JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f"analysis_results_{timestamp}.json"
        
        # Convert to JSON-serializable format
        json_results = json.dumps(all_results, indent=2, default=str)
        
        with open(filename, 'w') as f:
            f.write(json_results)
        
        print(f"✓ JSON results saved to: {filename}")
        
        return str(filename)

if __name__ == "__main__":
    print("Testing Report Generator...")
    print("=" * 60)
    
    # Create sample results
    test_data_quality = {
        'Gold': {'status': 'ok', 'records': 1000, 'date_range': {'start': '2020-01-01', 'end': '2024-12-31'},
                'completeness': 95.5, 'latest_price': 1950.00, 'latest_date': '2024-12-31'}
    }
    
    test_analysis = {
        'statistical': {
            'Gold': {'mean': 1900, 'median': 1850, 'std': 150, 'min': 1500, 'max': 2200, 'cv': 7.89, 'skewness': 0.5, 'kurtosis': -0.3}
        }
    }
    
    generator = ReportGenerator("./test_reports")
    report = generator.generate_full_report(test_data_quality, test_analysis, {}, {})
    
    print("\n✓ Test report generated")
    print("=" * 60)