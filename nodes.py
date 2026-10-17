import os
import json
import io
import subprocess
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from firecrawl import FirecrawlApp
from schemas import AgentState, CandidateProfile, JobMatchScore

def get_gemini_llm():
    """Returns a primary Gemini LLM instance."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def get_hf_llm(state: AgentState):
    """Returns a serverless Hugging Face chat context instance as a fallback option."""
    token = state.hf_token if state.hf_token else st.secrets.get("HF_TOKEN", "")
    if token:
        base_llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
            task="conversational",
            temperature=0.1,
            huggingfacehub_api_token=token,
            max_new_tokens=2048
        )
        return ChatHuggingFace(llm=base_llm)
    raise RuntimeError("Hugging Face API token is missing inside your configurations.")

def extract_profile_node(state: AgentState):
    """Parses raw text data to build structured candidate profile metadata blocks."""
    prompt = f"Extract a clean candidate profile matrix from this raw CV text content:\n\n{state.cv_text}"
    google_key = state.google_api_key if state.google_api_key else st.secrets.get("GOOGLE_API_KEY", "")
    
    if google_key:
        try:
            llm = get_gemini_llm()
            structured_gemini = llm.with_structured_output(CandidateProfile)
            extracted_data = structured_gemini.invoke(prompt)
            return {"profile": extracted_data}
        except Exception as e:
            print(f"⚠️ Gemini execution failed ({e}). Re-routing to fallback...")

    llm = get_hf_llm(state)
    messages = [
        SystemMessage(content=(
            "You are a strict data-formatting AI tool. You must respond ONLY with a valid raw JSON object matching this schema exactly:\n"
            "{\"name\": \"string\", \"skills\": [\"string\"], \"experience_summary\": \"string\"}\n"
            "Do not include markdown tags like ```json, preamble text, or explanations."
        )),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = llm.invoke(messages).content.strip()
        cleaned_json = response.split("```json")[-1].split("```").strip() if "```" in response else response
        data = json.loads(cleaned_json)
        return {"profile": CandidateProfile(
            name=data.get("name", "Unknown Candidate"),
            skills=data.get("skills", []),
            experience_summary=data.get("experience_summary", "N/A")
        )}
    except Exception as fallback_err:
        print(f"❌ Fallback processing failed: {fallback_err}")
        return {"profile": CandidateProfile(name="Backup Candidate Extraction", skills=["Python"], experience_summary="Processed via fallback infrastructure.")}

def generate_query_node(state: AgentState):
    """Generates optimized job-board query terms from candidate attributes."""
    prompt = (
        f"You are an expert recruiter. Read this candidate's background and skills:\n"
        f"Background: {state.profile.experience_summary}\n"
        f"Skills: {state.profile.skills}\n\n"
        f"Based strictly on their profile, output a single target job title phrase "
        f"(1 to 3 words max) to search for on live job boards. "
        f"Do not output markdown, quotes, or extra text. Output just the plain title phrase."
    )
    google_key = state.google_api_key if state.google_api_key else st.secrets.get("GOOGLE_API_KEY", "")
    
    if google_key:
        try:
            llm = get_gemini_llm()
            query_response = llm.invoke(prompt)
            return {"search_query": query_response.content.strip().replace("'", "").replace('"', "")}
        except Exception as e:
            print(f"⚠️ Gemini context construction failed ({e}). Re-routing to fallback...")

    llm = get_hf_llm(state)
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages).content.strip()
    clean_query = response.replace("'", "").replace('"', "").split("\n")[0].strip()
    return {"search_query": clean_query}

def fetch_jobs_node(state: AgentState):
    """Queries live data platforms using Firecrawl search configurations."""
    search_keyword = state.search_query.strip() if state.search_query else "AI Scientist"
    time_scope = getattr(state, 'time_filter', 'past 3 days')
    firecrawl_key = state.firecrawl_api_key if state.firecrawl_api_key else st.secrets.get("FIRECRAWL_API_KEY", "")
    
    if firecrawl_key:
        try:
            app = FirecrawlApp(api_key=firecrawl_key)
            search_query_string = f'"{search_keyword}" remote jobs hiring {time_scope}'
            search_result = app.search(query=search_query_string, params={"limit": 3})
            
            processed_jobs = []
            if search_result and 'data' in search_result:
                for idx, item in enumerate(search_result['data']):
                    processed_jobs.append({
                        "id": f"firecrawl_{idx}",
                        "title": item.get("title", f"{search_keyword} Role"),
                        "requirements": f"Source: {item.get('url')}. Content Snippet: {item.get('markdown', '')[:1000]}"
                    })
            if len(processed_jobs) > 0:
                return {"raw_jobs": processed_jobs}
        except Exception as e:
            print(f"❌ Firecrawl script failure: {e}. Utilizing fallback sample entries.")
        
    return {"raw_jobs": [
        {"id": "job_01", "title": f"Senior {search_keyword} Specialist", "requirements": f"Requires Python expertise, system architecture design, and advanced financial analytics workflows verified within the {time_scope} window."},
        {"id": "job_02", "title": f"FinTech {search_keyword} Lead", "requirements": f"Requires extensive mastery deploying machine learning pipelines, fine-tuning corporate models, and engineering robust RAG architectures within the {time_scope} window."}
    ]}
def rank_jobs_node(state: AgentState):
    """Evaluates suitability indexes and creates the structured shortlist."""
    rankings = []
    shortlist = []
    current_threshold = getattr(state, 'match_threshold', 70)
    google_key = state.google_api_key if state.google_api_key else st.secrets.get("GOOGLE_API_KEY", "")
    
    for job in state.raw_jobs:
        prompt = f"""
        Compare Candidate Skills: {state.profile.skills}
        Against Job Title: {job['title']}
        Job Data context: {job['requirements']}
        Calculate a matching accuracy score from 0 to 100. Provide a detailed critique explanation on what skills the candidate is completely missing for this job.
        """
        evaluated = False
        if google_key:
            try:
                llm = get_gemini_llm()
                structured_ranker = llm.with_structured_output(JobMatchScore)
                score_card = structured_ranker.invoke(prompt)
                score_card.job_id = job['id']
                score_card.job_title = job['title']
                score_card.threshold_passed = True if score_card.fit_score >= current_threshold else False
                rankings.append(score_card)
                if score_card.threshold_passed:
                    shortlist.append(score_card)
                evaluated = True
            except Exception as e:
                print(f"⚠️ Gemini ranking failed for {job['title']}. Routing to fallback...")

        if not evaluated:
            try:
                llm = get_hf_llm(state)
                messages = [
                    SystemMessage(content=(
                        "You are a strict evaluator. Output ONLY a valid raw JSON object matching this structure exactly:\n"
                        "{\"fit_score\": 85, \"gap_explanation\": \"Missing explicit details\"}\n"
                        "Do not include markdown tags like ```json or trailing text symbols."
                    )),
                    HumanMessage(content=prompt)
                ]
                response = llm.invoke(messages).content.strip()
                cleaned_json = response.split("```json")[-1].split("```").strip() if "```" in response else response
                data = json.loads(cleaned_json)
                score = int(data.get("fit_score", 50))
                score_card = JobMatchScore(
                    job_id=job['id'],
                    job_title=job['title'],
                    fit_score=score,
                    gap_explanation=data.get("gap_explanation", "Evaluated via text pipeline alternative."),
                    threshold_passed=True if score >= current_threshold else False
                )
                rankings.append(score_card)
                if score_card.threshold_passed:
                    shortlist.append(score_card)
            except Exception as rank_err:
                print(f"⚠️ Fallback evaluation exception: {rank_err}")
                
    shortlist.sort(key=lambda x: x.fit_score, reverse=True)
    return {"ranked_jobs": rankings, "top_shortlist": shortlist}

def drafter_agent_node(state: AgentState):
    """Drafter Agent: Formulates customized application assets using LaTeX typography syntax."""
    if not state.top_shortlist:
        return {"drafted_latex": "", "approved": True}
        
    target_job = state.top_shortlist[state.current_job_idx]
    feedback_context = f"Previous execution review errors to address:\n{state.review_feedback}" if state.review_feedback else ""
    
    prompt = f"""
    You are a professional executive typography designer. Write a highly tailored cover letter 
    for {state.profile.name} targeting the role of {target_job.job_title}.
    
    Candidate Summary: {state.profile.experience_summary}
    Core Competencies: {state.profile.skills}
    {feedback_context}
    Return ONLY valid, raw, compilation-ready LaTeX syntax code. 
    Include standard document structural markers like '\\documentclass{{article}}\(', '\begin{{document}}', and '\end\){{document}}'.
    Ensure that special reserved layout characters like %, &, \$, and _ are properly escaped.
    Do NOT wrap code segments within markdown block structures like ```latex. Begin immediately with '\\documentclass'.
    """
    
    google_key = state.google_api_key if state.google_api_key else st.secrets.get("GOOGLE_API_KEY", "")
    if google_key:
        response = get_gemini_llm().invoke(prompt).content.strip()
    else:
        response = get_hf_llm(state).invoke([HumanMessage(content=prompt)]).content.strip()
        
    if response.startswith("```"):
        response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return {"drafted_latex": response}
def reviewer_agent_node(state: AgentState):
    """Reviewer Agent: Checks text layout outputs against system rules criteria."""
    if not state.drafted_latex:
        return {"approved": True}
        
    prompt = f"""
    You are a critical quality control inspector checking LaTeX documents for compile defects.
    Analyze the following code payload for syntax formatting bugs or missing structures:
    
    {state.drafted_latex}
    
    If you discover unescaped characters, incomplete sections, or bad layout properties, 
    provide actionable correction details. 
    Otherwise, if the script is complete, flawless, and compilation-ready, reply with EXACTLY the word: APPROVED
    """
    
    google_key = state.google_api_key if state.google_api_key else st.secrets.get("GOOGLE_API_KEY", "")
    if google_key:
        feedback = get_gemini_llm().invoke(prompt).content.strip()
    else:
        feedback = get_hf_llm(state).invoke([HumanMessage(content=prompt)]).content.strip()
        
    if "APPROVED" in feedback.upper():
        return {"approved": True, "review_feedback": ""}
    return {"approved": False, "review_feedback": feedback}

def compile_latex_node(state: AgentState):
    """LaTeX to PDF Engine: Compiles structural documents into standard binary assets via CLI."""
    if not state.drafted_latex:
        return {"application_pdf_binary": None}
        
    tex_filename = "tailored_application.tex"
    pdf_filename = "tailored_application.pdf"
    
    with open(tex_filename, "w", encoding="utf-8") as f:
        f.write(state.drafted_latex)
        
    try:
        # Note: 'pdflatex' can be substituted here based on your environment's path variables
        subprocess.run(["tectonic", tex_filename], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        with open(pdf_filename, "rb") as f:
            pdf_bytes = f.read()
            
        # Clean up temporary generation artifacts
        if os.path.exists(tex_filename): os.remove(tex_filename)
        if os.path.exists(pdf_filename): os.remove(pdf_filename)
        
        return {"application_pdf_binary": pdf_bytes, "approved": True}
    except Exception as e:
        print(f"💥 PDF Engine Compiling Exception: {e}")
        return {"approved": False, "review_feedback": f"System compilation failed: {str(e)}. Please correct syntax bugs."}
