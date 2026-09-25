import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from firecrawl import FirecrawlApp
from schemas import AgentState, CandidateProfile, JobMatchScore

def get_llm_fallback(state: AgentState):
    """
    Tries to instantiate the primary Gemini LLM framework. 
    If it fails or the Google key is missing, it dynamically switches 
    to a Hugging Face open-source inference endpoint model.
    """
    # Strategy 1: Attempt running on premium Google Gemini Infrastructure
    if state.google_api_key and os.environ.get("GOOGLE_API_KEY"):
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            return llm, "gemini"
        except Exception as e:
            print(f"⚠️ Gemini Initialization failed: {e}. Attempting Hugging Face fallback...")

    # Strategy 2: Dynamic fallback execution space on Hugging Face Serverless Endpoints
    if state.hf_token and os.environ.get("HF_TOKEN"):
        try:
            llm = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.1-8B-Instruct",
                task="text-generation",
                temperature=0.1,
                huggingfacehub_api_token=state.hf_token
            )
            return llm, "huggingface"
        except Exception as e:
            print(f"❌ Hugging Face Fallback endpoint failed to spin up: {e}")
            
    raise RuntimeError("Critical Pipeline Disruption: No active or working LLM credentials available.")

def extract_profile_node(state: AgentState):
    llm, provider = get_llm_fallback(state)
    prompt = f"Extract a clean candidate profile matrix from this raw CV text content:\n\n{state.cv_text}"
    
    if provider == "gemini":
        structured_gemini = llm.with_structured_output(CandidateProfile)
        extracted_data = structured_gemini.invoke(prompt)
        return {"profile": extracted_data}
    else:
        # Structured JSON extraction helper code block tailored for Llama endpoints
        hf_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"You are a strict data-formatting AI tool. You must respond ONLY with a valid raw JSON object matching this schema:\n"
            f"{{\"name\": \"string\", \"skills\": [\"string\"], \"experience_summary\": \"string\"}}\n"
            f"Do not include markdown tags, preamble text, or explanations.<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
        response = llm.invoke(hf_prompt).strip()
        try:
            cleaned_json = response.split("```json")[-1].split("```")[0].strip() if "```" in response else response
            data = json.loads(cleaned_json)
            return {"profile": CandidateProfile(
                name=data.get("name", "Unknown Candidate"),
                skills=data.get("skills", []),
                experience_summary=data.get("experience_summary", "N/A")
            )}
        except Exception:
            return {"profile": CandidateProfile(name="Backup Candidate Extraction", skills=["Python"], experience_summary="Extracted via fallback model line.")}

def generate_query_node(state: AgentState):
    llm, provider = get_llm_fallback(state)
    prompt = (
        f"You are an expert recruiter. Read this candidate's background and skills:\n"
        f"Background: {state.profile.experience_summary}\n"
        f"Skills: {state.profile.skills}\n\n"
        f"Based strictly on their profile, output a single target job title phrase "
        f"(1 to 3 words max) to search for on live job boards. "
        f"Do not output markdown, quotes, or extra text. Output just the plain title phrase."
    )
    
    if provider == "gemini":
        query_response = llm.invoke(prompt)
        clean_query = query_response.content.strip().replace("'", "").replace('"', "")
    else:
        hf_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        response = llm.invoke(hf_prompt)
        clean_query = response.strip().replace("'", "").replace('"', "").split("\n")[0]
        
    return {"search_query": clean_query}

def fetch_jobs_node(state: AgentState):
    search_keyword = state.search_query.strip() if state.search_query else "developer"
    
    if state.firecrawl_api_key:
        try:
            app = FirecrawlApp(api_key=state.firecrawl_api_key)
            search_result = app.search(
                query=f"site:://remoteok.com{search_keyword}-jobs",
                params={"limit": 3}
            )
            
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
            print(f"❌ Firecrawl Extraction failed: {e}. Switching to default sample listings.")
        
    return {"raw_jobs": [
        {"id": "fallback_01", "title": f"Senior {search_keyword} Engineer", "requirements": "Requires Python expertise, system architecture design, and LLM orchestration workflows."},
        {"id": "fallback_02", "title": f"AI Solutions Specialist", "requirements": "Requires experience deploying machine learning pipelines, fine-tuning, and RAG architectures."}
    ]}

def rank_jobs_node(state: AgentState):
    llm, provider = get_llm_fallback(state)
    rankings = []
    current_threshold = getattr(state, 'match_threshold', 70)
    
    for job in state.raw_jobs:
        prompt = f"""
        Compare Candidate Skills: {state.profile.skills}
        Against Job Title: {job['title']}
        Job Data context: {job['requirements']}
        Calculate a matching accuracy score from 0 to 100. Provide a detailed critique explanation on what skills the candidate is completely missing for this job.
        """
        
        if provider == "gemini":
            try:
                structured_ranker = llm.with_structured_output(JobMatchScore)
                score_card = structured_ranker.invoke(prompt)
                score_card.job_id = job['id']
                score_card.job_title = job['title']
                score_card.threshold_passed = True if score_card.fit_score >= current_threshold else False
                rankings.append(score_card)
            except Exception as e:
                print(f"Evaluation error: {e}")
        else:
            hf_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"You are an evaluator. Output ONLY valid raw JSON matching this structure exactly:\n"
                f"{{\"fit_score\": 85, \"gap_explanation\": \"Missing explicit documentation detail rules\"}}\n"
                f"Do not write markdown tags.<|eot_id|>\n"
                f"<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>\n"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            )
            try:
                response = llm.invoke(hf_prompt).strip()
                cleaned_json = response.split("```json")[-1].split("```")[0].strip() if "```" in response else response
                data = json.loads(cleaned_json)
                score = int(data.get("fit_score", 50))
                rankings.append(JobMatchScore(
                    job_id=job['id'],
                    job_title=job['title'],
                    fit_score=score,
                    gap_explanation=data.get("gap_explanation", "Evaluated via text pipeline alternative."),
                    threshold_passed=True if score >= current_threshold else False
                ))
            except Exception:
                rankings.append(JobMatchScore(job_id=job['id'], job_title=job['title'], fit_score=60, gap_explanation="Fallback parser default.", threshold_passed=False))
                
    return {"ranked_jobs": rankings}

def generate_cover_letters_node(state: AgentState):
    llm, provider = get_llm_fallback(state)
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
            Structure it cleanly with dates, salutations, body paragraph, and closing. Do not output markdown text styling tags.
            """
            
            if provider == "gemini":
                draft = llm.invoke(writer_prompt).content.strip()
            else:
                hf_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{writer_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                draft = llm.invoke(hf_prompt).strip()
                
            drafted_letters[score_card.job_id] = draft
            
    return {"cover_letters": drafted_letters}
