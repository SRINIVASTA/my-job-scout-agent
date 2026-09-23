from langgraph.graph import StateGraph, START, END
from schemas import AgentState
from nodes import extract_profile_node, generate_query_node, fetch_jobs_node, rank_jobs_node, generate_cover_letters_node

def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("extract_profile", extract_profile_node)
    workflow.add_node("generate_query", generate_query_node)
    workflow.add_node("fetch_jobs", fetch_jobs_node)
    workflow.add_node("rank_jobs", rank_jobs_node)
    workflow.add_node("generate_cover_letters", generate_cover_letters_node)

    workflow.add_edge(START, "extract_profile")
    workflow.add_edge("extract_profile", "generate_query")
    workflow.add_edge("generate_query", "fetch_jobs")
    workflow.add_edge("fetch_jobs", "rank_jobs")
    workflow.add_edge("rank_jobs", "generate_cover_letters")
    workflow.add_edge("generate_cover_letters", END)

    return workflow.compile()

agent_app = build_workflow()
