"""
Deep Agent Module
LangGraph-based agent for commodity analysis using OpenAI
"""
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from config import Config
from state import AgentState, create_initial_state, log_execution, mark_task_complete
from data_collector import DataCollector
from analysis_tools import CommodityAnalyzer
from forecasting import CommodityForecaster
from risk_analysis import RiskAnalyzer
from visualization import Visualizer
from report_generator import ReportGenerator
import warnings
warnings.filterwarnings('ignore')

class CommodityAnalysisAgent:
    """Deep Agent for comprehensive commodity analysis"""
    
    def __init__(self, llm_model: str = "gpt-4"):
        self.llm_model = llm_model
        
        # Initialize LLM with OpenAI
        if Config.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model=self.llm_model,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
                openai_api_key=Config.OPENAI_API_KEY
            )
        else:
            self.llm = None
            print("WARNING: No OpenAI API key found. LLM features will be limited.")
        
        # Initialize components
        self.data_collector = DataCollector()
        self.analyzer = CommodityAnalyzer()
        self.forecaster = CommodityForecaster()
        self.risk_analyzer = RiskAnalyzer()
        self.visualizer = Visualizer()
        self.report_generator = ReportGenerator()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("data_collector", self.data_collection_node)
        workflow.add_node("analyzer", self.analysis_node)
        workflow.add_node("forecaster", self.forecasting_node)
        workflow.add_node("risk_assessor", self.risk_assessment_node)
        workflow.add_node("visualizer", self.visualization_node)
        workflow.add_node("reporter", self.reporting_node)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Add edges
        workflow.add_edge("planner", "data_collector")
        workflow.add_edge("data_collector", "analyzer")
        workflow.add_edge("analyzer", "forecaster")
        workflow.add_edge("forecaster", "risk_assessor")
        workflow.add_edge("risk_assessor", "visualizer")
        workflow.add_edge("visualizer", "reporter")
        workflow.add_edge("reporter", END)
        
        return workflow.compile()
    
    def planner_node(self, state: AgentState) -> AgentState:
        """Planning node - creates analysis plan"""
        print("\n[PLANNER] Creating analysis plan...")
        
        state = log_execution(state, "planner", "Starting analysis planning", "info")
        
        # Create task plan
        tasks = [
            {"name": "data_collection", "description": "Fetch commodity price data"},
            {"name": "statistical_analysis", "description": "Perform statistical analysis"},
            {"name": "trend_analysis", "description": "Identify price trends"},
            {"name": "forecasting", "description": "Generate price forecasts"},
            {"name": "risk_assessment", "description": "Assess risks and volatility"},
            {"name": "visualization", "description": "Create charts and graphs"},
            {"name": "reporting", "description": "Generate final report"}
        ]
        
        state["task_plan"] = tasks
        state["pending_tasks"] = [t["name"] for t in tasks]
        state["current_task"] = "data_collection"
        
        # Add to context
        state["analysis_context"].append(
            f"Analyzing {len(state['commodities_to_analyze'])} commodities: " +
            ", ".join(state['commodities_to_analyze'])
        )
        
        state = log_execution(state, "planner", f"Created plan with {len(tasks)} tasks", "success")
        
        return state
    
    def data_collection_node(self, state: AgentState) -> AgentState:
        """Data collection node - fetches commodity data"""
        print("\n[DATA COLLECTOR] Fetching commodity data...")
        
        state = log_execution(state, "data_collector", "Starting data collection", "info")
        state["current_task"] = "data_collection"
        
        try:
            # Fetch data for all commodities
            all_data = self.data_collector.fetch_all_commodities()
            
            if not all_data:
                state = log_execution(state, "data_collector", "No data collected", "error")
                state["errors"].append("Failed to collect commodity data")
                return state
            
            state["raw_data"] = all_data
            state["data_collection_complete"] = True
            
            # Generate data quality report
            quality_report = self.data_collector.get_data_quality_report(all_data)
            state["data_quality_report"] = quality_report
            
            state = mark_task_complete(state, "data_collection")
            state = log_execution(
                state, "data_collector", 
                f"Collected data for {len(all_data)} commodities", 
                "success"
            )
            
        except Exception as e:
            state = log_execution(state, "data_collector", f"Error: {str(e)}", "error")
            state["errors"].append(f"Data collection error: {str(e)}")
        
        return state
    
    def analysis_node(self, state: AgentState) -> AgentState:
        """Analysis node - performs statistical and trend analysis"""
        print("\n[ANALYZER] Performing analysis...")
        
        state = log_execution(state, "analyzer", "Starting analysis", "info")
        state["current_task"] = "analysis"
        
        try:
            data_dict = state["raw_data"]
            
            if not data_dict:
                state = log_execution(state, "analyzer", "No data available for analysis", "error")
                return state
            
            # Perform all analyses
            analysis_results = self.analyzer.analyze_all(data_dict)
            
            state["statistical_analysis"] = analysis_results.get("statistical", {})
            state["trend_analysis"] = analysis_results.get("trends", {})
            state["pattern_detection"] = analysis_results.get("patterns", {})
            state["correlation_analysis"] = analysis_results.get("correlations", {})
            
            # Extract key insights
            for commodity, trend in analysis_results.get("trends", {}).items():
                if "error" not in trend:
                    direction = trend.get("trend_direction", "neutral")
                    state["key_insights"].append(
                        f"{commodity}: {direction} trend with strength {trend.get('trend_strength', 0):.2f}"
                    )
            
            state = mark_task_complete(state, "statistical_analysis")
            state = mark_task_complete(state, "trend_analysis")
            state = log_execution(state, "analyzer", "Analysis completed", "success")
            
        except Exception as e:
            state = log_execution(state, "analyzer", f"Error: {str(e)}", "error")
            state["errors"].append(f"Analysis error: {str(e)}")
        
        return state
    
    def forecasting_node(self, state: AgentState) -> AgentState:
        """Forecasting node - generates price forecasts"""
        print("\n[FORECASTER] Generating forecasts...")
        
        state = log_execution(state, "forecaster", "Starting forecasting", "info")
        state["current_task"] = "forecasting"
        
        try:
            data_dict = state["raw_data"]
            
            if not data_dict:
                state = log_execution(state, "forecaster", "No data for forecasting", "error")
                return state
            
            # Generate forecasts
            forecast_results = self.forecaster.forecast_all(data_dict, method="ensemble")
            
            state["forecast_results"] = forecast_results
            
            # Calculate confidence scores
            confidence = {}
            for commodity, forecast in forecast_results.items():
                if "error" not in forecast and "summary" in forecast:
                    confidence[commodity] = forecast["summary"].get("confidence_score", 0.0)
            
            state["forecast_confidence"] = confidence
            
            # Add insights
            for commodity, forecast in forecast_results.items():
                if "error" not in forecast:
                    change = forecast.get("expected_change", 0)
                    state["key_insights"].append(
                        f"{commodity} forecast: {change:+.2f}% expected change"
                    )
            
            state = mark_task_complete(state, "forecasting")
            state = log_execution(state, "forecaster", "Forecasting completed", "success")
            
        except Exception as e:
            state = log_execution(state, "forecaster", f"Error: {str(e)}", "error")
            state["errors"].append(f"Forecasting error: {str(e)}")
        
        return state
    
    def risk_assessment_node(self, state: AgentState) -> AgentState:
        """Risk assessment node - analyzes risks"""
        print("\n[RISK ASSESSOR] Assessing risks...")
        
        state = log_execution(state, "risk_assessor", "Starting risk assessment", "info")
        state["current_task"] = "risk_assessment"
        
        try:
            data_dict = state["raw_data"]
            
            if not data_dict:
                state = log_execution(state, "risk_assessor", "No data for risk assessment", "error")
                return state
            
            # Perform risk analysis
            risk_results = self.risk_analyzer.analyze_all_risks(data_dict)
            
            state["risk_metrics"] = risk_results
            
            # Extract VaR values
            var_results = {}
            for commodity, metrics in risk_results.get("individual", {}).items():
                if "var" in metrics and "error" not in metrics["var"]:
                    var_results[commodity] = metrics["var"]["var_historical"]["percentage"]
            
            state["var_results"] = var_results
            
            # Add risk insights
            for commodity, var_pct in var_results.items():
                state["key_insights"].append(
                    f"{commodity} VaR: {abs(var_pct):.2f}% daily risk at 95% confidence"
                )
            
            state = mark_task_complete(state, "risk_assessment")
            state = log_execution(state, "risk_assessor", "Risk assessment completed", "success")
            
        except Exception as e:
            state = log_execution(state, "risk_assessor", f"Error: {str(e)}", "error")
            state["errors"].append(f"Risk assessment error: {str(e)}")
        
        return state
    
    def visualization_node(self, state: AgentState) -> AgentState:
        """Visualization node - creates charts"""
        print("\n[VISUALIZER] Creating visualizations...")
        
        state = log_execution(state, "visualizer", "Starting visualization", "info")
        state["current_task"] = "visualization"
        
        try:
            data_dict = state["raw_data"]
            forecast_results = state.get("forecast_results", {})
            risk_results = state.get("risk_metrics", {})
            
            # Create all visualizations
            charts = self.visualizer.create_all_visualizations(
                data_dict, 
                forecast_results, 
                risk_results
            )
            
            state["visualizations"] = charts
            
            state = mark_task_complete(state, "visualization")
            state = log_execution(
                state, "visualizer", 
                f"Created {len(charts)} visualizations", 
                "success"
            )
            
        except Exception as e:
            state = log_execution(state, "visualizer", f"Error: {str(e)}", "error")
            state["errors"].append(f"Visualization error: {str(e)}")
        
        return state
    
    def reporting_node(self, state: AgentState) -> AgentState:
        """Reporting node - generates final report"""
        print("\n[REPORTER] Generating report...")
        
        state = log_execution(state, "reporter", "Starting report generation", "info")
        state["current_task"] = "reporting"
        
        try:
            # Compile all results
            analysis_results = {
                "statistical": state.get("statistical_analysis", {}),
                "trends": state.get("trend_analysis", {}),
                "patterns": state.get("pattern_detection", {}),
                "correlations": state.get("correlation_analysis", {})
            }
            
            # Generate report
            report = self.report_generator.generate_full_report(
                data_quality=state.get("data_quality_report", {}),
                analysis_results=analysis_results,
                forecast_results=state.get("forecast_results", {}),
                risk_results=state.get("risk_metrics", {}),
                charts=state.get("visualizations", [])
            )
            
            state["final_report"] = report
            
            # Save JSON results
            all_results = {
                "data_quality": state.get("data_quality_report", {}),
                "analysis": analysis_results,
                "forecasts": state.get("forecast_results", {}),
                "risk": state.get("risk_metrics", {}),
                "insights": state.get("key_insights", [])
            }
            self.report_generator.save_json_results(all_results)
            
            state = mark_task_complete(state, "reporting")
            state = log_execution(state, "reporter", "Report generation completed", "success")
            
            state["should_continue"] = False
            
        except Exception as e:
            state = log_execution(state, "reporter", f"Error: {str(e)}", "error")
            state["errors"].append(f"Reporting error: {str(e)}")
        
        return state
    
    async def run_analysis(self, user_query: str, commodities: List[str] = None) -> Dict[str, Any]:
        """Run the complete analysis workflow"""
        
        # Create initial state
        initial_state = create_initial_state(user_query, commodities)
        
        print("\n" + "=" * 80)
        print("STARTING COMMODITY ANALYSIS AGENT")
        print("=" * 80)
        
        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Prepare results
            results = {
                "success": len(final_state["errors"]) == 0,
                "completed_tasks": final_state["completed_tasks"],
                "key_insights": final_state["key_insights"],
                "warnings": final_state["warnings"],
                "errors": final_state["errors"],
                "report": final_state.get("final_report"),
                "visualizations": final_state.get("visualizations", [])
            }
            
            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETE")
            print("=" * 80)
            print(f"✓ Completed {len(final_state['completed_tasks'])} tasks")
            print(f"✓ Generated {len(final_state['key_insights'])} insights")
            print(f"✓ Created {len(final_state.get('visualizations', []))} visualizations")
            
            if final_state["errors"]:
                print(f"⚠ {len(final_state['errors'])} errors occurred")
            
            return results
            
        except Exception as e:
            print(f"\n✗ Analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "completed_tasks": [],
                "key_insights": [],
                "warnings": [],
                "errors": [str(e)]
            }

# Test function
if __name__ == "__main__":
    import asyncio
    
    print("Testing Commodity Analysis Agent...")
    print("=" * 60)
    
    agent = CommodityAnalysisAgent(llm_model="gpt-4")
    
    query = "Analyze Gold and Crude Oil prices with forecasts"
    
    async def test():
        results = await agent.run_analysis(query, commodities=["Gold", "Crude_Oil"])
        print(f"\nTest completed: {'Success' if results['success'] else 'Failed'}")
    
    asyncio.run(test())
    
    print("=" * 60)