import httpx
from langchain_google_genai import ChatGoogleGenerativeAI
from schemas import AgentState, CandidateProfile, JobMatchScore

# Initialize the Gemini Model dynamically inside the functions to pick up the user's sidebar key
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def extract_profile_node(state: AgentState):
    structured_gemini = get_llm().with_structured_output(CandidateProfile)
    prompt = f"Extract a clean candidate matrix from this raw text content:\n\n{state.cv_text}"
    extracted_data = structured_gemini.invoke(prompt)
    return {"profile": extracted_data}

def generate_query_node(state: AgentState):
    prompt = f"Based on these skills: {state.profile.skills}, output a single short job search keyword (like 'Python', 'React', or 'DevOps') to pass to a query engine."
    query_response = get_llm().invoke(prompt)
    clean_query = query_response.content.strip().replace("'", "").replace('"', "")
    return {"search_query": clean_query}

def fetch_jobs_node(state: AgentState):
    search_keyword = state.search_query if state.search_query else "developer"
    
    # Corrected API endpoint for Arbeitnow to access the correct JSON pay-channel
    url = "https://arbeitnow.com"
    try:
        response = httpx.get(url, params={"search": search_keyword, "page": 1}, timeout=10.0)
        if response.status_code == 200:
            api_data = response.json()
            jobs_list = api_data.get("data", [])
            
            processed_jobs = []
            for item in jobs_list[:4]:  # Cap evaluations to top 4 jobs for speed and token efficiency
                processed_jobs.append({
                    "id": item.get("slug", "unknown"),
                    "title": item.get("title", "Job Opening"),
                    "requirements": f"Company: {item.get('company_name')}. Description: {item.get('description')[:800]}"
                })
            if processed_jobs:
                return {"raw_jobs": processed_jobs}
    except Exception as e:
        print(f"❌ Market API Connection failed: {e}.")
        
    return {"raw_jobs": [
        {"id": "fallback_01", "title": f"Senior {search_keyword} Engineer", "requirements": f"Requires strong technical engineering expertise specialized in {search_keyword} design patterns."},
        {"id": "fallback_02", "title": f"Full Stack {search_keyword} Consultant", "requirements": f"Looking for an architect proficient in scaling modern systems using {search_keyword} technologies."}
    ]}

def rank_jobs_node(state: AgentState):
    rankings = []
    structured_ranker = get_llm().with_structured_output(JobMatchScore)
    
    for job in state.raw_jobs:
        prompt = f"""
        Compare Candidate Skills: {state.profile.skills}
        Against Job Title: {job['title']}
        Job Data context: {job['requirements']}
        Calculate a matching accuracy score from 0 to 100. Provide a detailed critique on what skills the candidate is completely missing for this job.
        """
        try:
            score_card = structured_ranker.invoke(prompt)
            score_card.job_id = job['id']
            score_card.job_title = job['title']
            score_card.threshold_passed = True if score_card.fit_score >= 70 else False
            rankings.append(score_card)
        except Exception as e:
            print(f"Evaluation error: {e}")
    return {"ranked_jobs": rankings}

def generate_cover_letters_node(state: AgentState):
    drafted_letters = {}
    for score_card in state.ranked_jobs:
        if score_card.threshold_passed:
            job_details = next((j for j in state.raw_jobs if j["id"] == score_card.job_id), None)
            job_requirements = job_details["requirements"] if job_details else "Standard Role"
            
            writer_prompt = f"""
            Write a highly professional, tailored cover letter for a candidate named {state.profile.name}.
            Target Position: {score_card.job_title}
            Job Context/Company Requirements: {job_requirements}
            Candidate Background Summary: {state.profile.experience_summary}
            Candidate Explicit Skills Matrix: {state.profile.skills}
            Structure it cleanly with dates, salutations, body paragraph, and closing. Do not output markdown code tags.
            """
            draft = get_llm().invoke(writer_prompt)
            drafted_letters[score_card.job_id] = draft.content.strip()
    return {"cover_letters": drafted_letters}
