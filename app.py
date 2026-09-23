import streamlit as st
import os
from typing import List, Optional, Dict, TypedDict
from pydantic import BaseModel, Field
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# --- 1. SECURE PASSWORD INPUT FOR DESKTOP / CLOUD ---
api_key = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.set_page_config(page_title="API Setup", page_icon="🔑")
    st.title("🔑 Google Gemini API Authentication")
    st.info("To run this desktop AI agent application, please enter your Google AI Studio API key below.")
    
    user_key = st.text_input("Enter your GOOGLE_API_KEY:", type="password", help="Get a free key from Google AI Studio")
    if user_key:
        os.environ["GOOGLE_API_KEY"] = user_key
        st.success("✅ Key loaded successfully! Refreshing app...")
        st.rerun()
    else:
        st.warning("⚠️ Please enter your API key password to unlock the dashboard.")
        st.stop() 
else:
    os.environ["GOOGLE_API_KEY"] = api_key

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

# --- 3. INITIALIZE COMPONENTS ---
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
        # Convert Pydantic object to string cleanly regardless of Pydantic v1 vs v2 configurations
        profile_json = profile.model_dump_json() if hasattr(profile, 'model_dump_json') else profile.json()
        prompt = f"Compare Job: {job['title']} at {job['company']} Description: {job['description']} with Profile: {profile_json}"
        assessment = structured_ranker.invoke(prompt)
        ranked.append(assessment)
    return {"ranked_jobs": ranked}

# --- 5. COMPILE THE WORKFLOW (Strictly Before Frontend Runs) ---
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

# Initializing global variable explicitly here
job_agent = workflow.compile()

# --- 6. STREAMLIT FRONTEND DASHBOARD ---
st.set_page_config(page_title="AI Job Scout", layout="wide", page_icon="🤖")
st.title("🤖 LangGraph AI Job Scout Agent")
st.caption("Powered by Google Gemini 2.5 & LangGraph")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Input Profile")
    input_method = st.radio("Choose how to provide your CV:", ["Upload PDF File", "Copy-Paste Text Input"])
    
    cv_text_to_process = ""
    
    if input_method == "Upload PDF File":
        uploaded_file = st.file_uploader("Drag and drop your CV profile here (PDF format):", type=["pdf"])
        if uploaded_file is not None:
            with st.spinner("📄 Reading PDF text data..."):
                try:
                    reader = PdfReader(uploaded_file)
                    extracted_text = ""
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            extracted_text += text + "\n"
                    
                    if extracted_text.strip():
                        cv_text_to_process = extracted_text
                        st.success(f"✅ Successfully extracted text from '{uploaded_file.name}'!")
                    else:
                        st.error("⚠️ Could not read text layers from this PDF.")
                except Exception as e:
                    st.error(f"❌ Failed to parse document: {e}")
            
            if cv_text_to_process:
                if st.checkbox("🔍 Preview Extracted Text"):
                    st.text_area("Extracted Payload Preview", value=cv_text_to_process[:1000], height=200, disabled=True)
                    
    else:
        cv_text_to_process = st.text_area(
            "Paste your CV text or professional summary below:", 
            height=300, 
            placeholder="John Doe\nSkills: Python, SQL..."
        )
        
    st.divider()
    run_btn = st.button("🚀 Run Analysis Agent", type="primary", use_container_width=True)

with col2:
    st.subheader("🎯 Agent Recommendations Dashboard")
    
    if run_btn:
        if not cv_text_to_process.strip():
            st.warning("⚠️ No CV content detected! Please upload a valid PDF or paste your profile details first.")
        else:
            with st.spinner("🧠 Agent is thinking... running LangGraph nodes..."):
                result = job_agent.invoke({"cv_text": cv_text_to_process})
                profile = result.get("profile")
                ranked_jobs = result.get("ranked_jobs", [])
                
                if profile:
                    st.success("✅ CV Parsed Successfully!")
                    st.write(f"**Identified Level:** `{profile.experience_level}`")
                    st.write(f"**Extracted Skills:** {', '.join(profile.skills)}")
                
                st.divider()
                
                for match in ranked_jobs:
                    with st.expander(f"🏢 **{match.job_title}** — {match.company} (Match Score: {match.match_score}%)", expanded=True):
                        st.progress(match.match_score / 100)
                        st.write(f"✅ **Your Matching Skills:** {', '.join(match.matching_skills)}")
                        st.write(f"❌ **Identified Gaps:** {', '.join(match.gaps)}")
                        st.info(f"💡 **Verdict:** {match.explanation}")
    else:
        st.info("Provide your CV data on the left panel and click run to activate the agent workflow loops.")
