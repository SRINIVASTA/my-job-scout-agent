from langgraph.graph import StateGraph, START, END
from schemas import AgentState
from nodes import (
    extract_profile_node, 
    generate_query_node, 
    fetch_jobs_node, 
    rank_jobs_node, 
    drafter_agent_node,
    reviewer_agent_node,
    compile_latex_node
)

def determine_next_step(state: AgentState):
    """Evaluates the structural approval flag state downstream."""
    if state.approved:
        return "compile_latex"
    return "drafter_agent"

def build_workflow():
    workflow = StateGraph(AgentState)

    # Core Pipeline Nodes
    workflow.add_node("extract_profile", extract_profile_node)
    workflow.add_node("generate_query", generate_query_node)
    workflow.add_node("fetch_jobs", fetch_jobs_node)
    workflow.add_node("rank_jobs", rank_jobs_node)
    
    # Target Application Agent Sub-Loops
    workflow.add_node("drafter_agent", drafter_agent_node)
    workflow.add_node("reviewer_agent", reviewer_agent_node)
    workflow.add_node("compile_latex", compile_latex_node)

    # Core Pipeline Transitions
    workflow.add_edge(START, "extract_profile")
    workflow.add_edge("extract_profile", "generate_query")
    workflow.add_edge("generate_query", "fetch_jobs")
    workflow.add_edge("fetch_jobs", "rank_jobs")
    workflow.add_edge("rank_jobs", "drafter_agent")
    
    # Agent Validation Sequence Loop Connection Points
    workflow.add_edge("drafter_agent", "reviewer_agent")
    workflow.add_conditional_edges(
        "reviewer_agent",
        determine_next_step,
        {
            "compile_latex": "compile_latex",
            "drafter_agent": "drafter_agent"
        }
    )
    workflow.add_edge("compile_latex", END)

    return workflow.compile()

agent_app = build_workflow()
