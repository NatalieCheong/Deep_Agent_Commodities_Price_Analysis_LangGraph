"""
State management for the Deep Agent system using LangGraph
"""
from typing import TypedDict, Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

class AgentState(TypedDict):
    """
    State object that flows through the LangGraph nodes
    Tracks all data, analysis results, and decisions made by the agent
    """
    # Input
    user_query: str
    commodities_to_analyze: List[str]
    
    # Data Collection
    raw_data: Dict[str, pd.DataFrame]
    data_quality_report: Dict[str, Any]
    data_collection_complete: bool
    
    # Analysis Results
    statistical_analysis: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    pattern_detection: Dict[str, Any]
    correlation_analysis: Dict[str, Any]
    
    # Forecasting
    forecast_results: Dict[str, Any]
    forecast_confidence: Dict[str, float]
    
    # Risk Assessment
    risk_metrics: Dict[str, Any]
    var_results: Dict[str, float]
    stress_test_results: Dict[str, Any]
    
    # Decision Making
    current_task: str
    completed_tasks: List[str]
    pending_tasks: List[str]
    task_plan: List[Dict[str, Any]]
    
    # Context & Memory
    analysis_context: List[str]
    key_insights: List[str]
    warnings: List[str]
    
    # Output
    final_report: Optional[str]
    visualizations: List[str]
    recommendations: List[str]
    
    # Metadata
    start_time: datetime
    end_time: Optional[datetime]
    execution_log: List[Dict[str, Any]]
    errors: List[str]
    
    # Agent Control
    should_continue: bool
    iteration_count: int
    max_iterations: int

def create_initial_state(user_query: str, commodities: List[str] = None) -> AgentState:
    """
    Create the initial state for the agent
    
    Args:
        user_query: The user's query/request
        commodities: List of commodities to analyze (if None, analyze all)
    
    Returns:
        Initialized AgentState
    """
    from config import Config
    
    if commodities is None:
        commodities = Config.get_commodity_list()
    
    return AgentState(
        # Input
        user_query=user_query,
        commodities_to_analyze=commodities,
        
        # Data Collection
        raw_data={},
        data_quality_report={},
        data_collection_complete=False,
        
        # Analysis Results
        statistical_analysis={},
        trend_analysis={},
        pattern_detection={},
        correlation_analysis={},
        
        # Forecasting
        forecast_results={},
        forecast_confidence={},
        
        # Risk Assessment
        risk_metrics={},
        var_results={},
        stress_test_results={},
        
        # Decision Making
        current_task="initialize",
        completed_tasks=[],
        pending_tasks=[],
        task_plan=[],
        
        # Context & Memory
        analysis_context=[],
        key_insights=[],
        warnings=[],
        
        # Output
        final_report=None,
        visualizations=[],
        recommendations=[],
        
        # Metadata
        start_time=datetime.now(),
        end_time=None,
        execution_log=[],
        errors=[],
        
        # Agent Control
        should_continue=True,
        iteration_count=0,
        max_iterations=50  # Safety limit
    )

def log_execution(state: AgentState, node_name: str, message: str, 
                 status: str = "info") -> AgentState:
    """
    Add an execution log entry to the state
    
    Args:
        state: Current agent state
        node_name: Name of the node logging
        message: Log message
        status: Log status (info, success, warning, error)
    
    Returns:
        Updated state
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "node": node_name,
        "message": message,
        "status": status,
        "iteration": state["iteration_count"]
    }
    
    state["execution_log"].append(log_entry)
    
    # Also add to appropriate list
    if status == "error":
        state["errors"].append(f"{node_name}: {message}")
    elif status == "warning":
        state["warnings"].append(f"{node_name}: {message}")
    
    return state

def add_insight(state: AgentState, insight: str) -> AgentState:
    """Add a key insight to the state"""
    if insight not in state["key_insights"]:
        state["key_insights"].append(insight)
    return state

def mark_task_complete(state: AgentState, task_name: str) -> AgentState:
    """Mark a task as completed"""
    if task_name not in state["completed_tasks"]:
        state["completed_tasks"].append(task_name)
    
    # Remove from pending if present
    if task_name in state["pending_tasks"]:
        state["pending_tasks"].remove(task_name)
    
    return state

def add_pending_task(state: AgentState, task_name: str) -> AgentState:
    """Add a task to pending list"""
    if task_name not in state["pending_tasks"] and task_name not in state["completed_tasks"]:
        state["pending_tasks"].append(task_name)
    return state

def get_summary(state: AgentState) -> Dict[str, Any]:
    """Get a summary of the current state"""
    return {
        "query": state["user_query"],
        "commodities": state["commodities_to_analyze"],
        "current_task": state["current_task"],
        "completed_tasks": len(state["completed_tasks"]),
        "pending_tasks": len(state["pending_tasks"]),
        "insights_found": len(state["key_insights"]),
        "warnings": len(state["warnings"]),
        "errors": len(state["errors"]),
        "iteration": state["iteration_count"],
        "data_collected": state["data_collection_complete"],
        "has_forecast": bool(state["forecast_results"]),
        "has_risk_analysis": bool(state["risk_metrics"])
    }