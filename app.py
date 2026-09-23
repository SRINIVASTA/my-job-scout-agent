import streamlit as st
import os
from typing import List, Optional, Dict, TypedDict
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# --- 1. SECURELY LOAD API KEY ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ GOOGLE_API_KEY missing! Please add it to your Streamlit Secrets panel.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- 2. DEFINE DATA SCHEMAS ---
class CVProfile(BaseModel):
    skills: List[str] = Field(description="List of technical and soft skills")
    experience_level: str = Field(description="Entry, Mid, Senior, or Lead level")
    preferred_roles: List[str] = Field(description="Target job titles based on history")

class JobMatchAssessment(BaseModel):
    job_title: str
    company: str
    match_score: int = Field(description="Score from 0 to 100 on profile fit")
    matching_skills: List[str] = Field(description="Skills that matched the CV")
    gaps: List[str] = Field(description="Required job skills missing from the CV")
    explanation: str = Field(description="Brief reason for this score")

class SimulatedJob(BaseModel):
    title: str
    company: str
    description: str

class SimulatedJobBoardResponse(BaseModel):
    jobs: List[SimulatedJob]

# --- 3. INITIALIZE INITIAL COMPONENTS ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class AgentState(TypedDict):
    cv_text: str
    profile: Optional[CVProfile]
    search_queries: List[str]
    raw_jobs: List[Dict]
    ranked_jobs: List[JobMatchAssessment]

# --- 4. BUILD LANGGRAPH AGENT NODES ---
def extract_cv_node(state: AgentState):
    structured_llm = llm.with_structured_output(CVProfile)
    profile = structured_llm.invoke(f"Extract details from this CV:\n\n{state['cv_text']}")
    return {"profile": profile}

def generate_queries_node(state: AgentState):
    profile = state["profile"]
    prompt = f"Create 2 search queries for a job board based on roles: {profile.preferred_roles}. Separate with commas."
    response = llm.invoke(prompt)
    queries = [q.strip() for q in response.content.split(",")]
    return {"search_queries": queries}

def fetch_jobs_node(state: AgentState):
    profile = state["profile"]
    structured_simulator = llm.with_structured_output(SimulatedJobBoardResponse)
    prompt = f"Generate 3 realistic tech job openings for these roles: {profile.preferred_roles}. Include matching skills from {profile.skills} and some skill gaps."
    simulated_response = structured_simulator.invoke(prompt)
    
    formatted_jobs = [{"title": j.title, "company": j.company, "description": j.description} for j in simulated_response.jobs]
    return {"raw_jobs": formatted_jobs}

def rank_jobs_node(state: AgentState):
    profile = state["profile"]
    raw_jobs = state["raw_jobs"]
    ranked = []
    structured_ranker = llm.with_structured_output(JobMatchAssessment)
    
    for job in raw_jobs:
        prompt = f"Compare Job: {job['title']} at {job['company']} Description: {job['description']} with Profile: {profile.model_dump_json()}"
        assessment = structured_ranker.invoke(prompt)
        ranked.append(assessment)
    return {"ranked_jobs": ranked}

# --- 5. COMPILE THE WORKFLOW ---
workflow = StateGraph(AgentState)
workflow.add_node("extract_cv", extract_cv_node)
workflow.add_node("generate_queries", generate_queries_node)
workflow.add_node("fetch_jobs", fetch_jobs_node)
workflow.add_node("rank_jobs", rank_jobs_node)

workflow.set_entry_point("extract_cv")
workflow.add_edge("extract_cv", "generate_queries")
workflow.add_edge("generate_queries", "fetch_jobs")
workflow.add_edge("fetch_jobs", "rank_jobs")
workflow.add_edge("rank_jobs", END)
job_agent = workflow.compile()

# --- 6. STREAMLIT FRONTEND USER INTERFACE ---
st.set_page_config(page_title="AI Job Scout", layout="wide", page_icon="🤖")
st.title("🤖 LangGraph AI Job Scout Agent")
st.caption("Powered by Google Gemini 2.5 & LangGraph — Securely Deployed via Streamlit Cloud")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Input Profile")
    cv_input = st.text_area(
        "Paste your CV text or professional summary below:", 
        height=300, 
        placeholder="John Doe\nSkills: Python, SQL..."
    )
    run_btn = st.button("🚀 Run Analysis Agent", type="primary", use_container_width=True)

with col2:
    st.subheader("🎯 Agent Recommendations Dashboard")
    
    if run_btn:
        if not cv_input.strip():
            st.warning("⚠️ Please provide some resume text first!")
        else:
            with st.spinner("🧠 Agent is thinking... running LangGraph nodes..."):
                # Run the LangGraph execution block
                result = job_agent.invoke({"cv_text": cv_input})
                profile = result.get("profile")
                ranked_jobs = result.get("ranked_jobs", [])
                
                # Show parsed skills metrics inside lookups
                if profile:
                    st.success("✅ CV Parsed Successfully!")
                    st.write(f"**Identified Level:** `{profile.experience_level}`")
                    st.write(f"**Extracted Skills:** {', '.join(profile.skills)}")
                
                st.divider()
                
                # Render scored opportunities
                for match in ranked_jobs:
                    with st.expander(f"🏢 **{match.job_title}** — {match.company} (Match Score: {match.match_score}%)", expanded=True):
                        st.progress(match.match_score / 100)
                        st.write(f"✅ **Your Matching Skills:** {', '.join(match.matching_skills)}")
                        st.write(f"❌ **Identified Gaps:** {', '.join(match.gaps)}")
                        st.info(f"💡 **Verdict:** {match.explanation}")
    else:
        st.info("Paste your CV on the left and click run to activate the agent workflow loops.")
