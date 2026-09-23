from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class CandidateProfile(BaseModel):
    name: str = Field(description="Full name of the candidate")
    skills: List[str] = Field(description="Core technical and soft skills extracted from the CV")
    experience_summary: str = Field(description="A brief summary of career background")

class JobMatchScore(BaseModel):
    job_id: str = Field(default="")
    job_title: str = Field(description="Title of the evaluated job position")
    fit_score: int = Field(description="A ranking fit score from 0 to 100")
    gap_explanation: str = Field(description="Honest explanation of gaps between candidate and job profile")
    threshold_passed: bool = Field(default=False, description="True if fit_score is >= 70")

class AgentState(BaseModel):
    cv_text: str = ""
    profile: Optional[CandidateProfile] = None
    search_query: str = ""
    raw_jobs: List[dict] = []
    ranked_jobs: List[JobMatchScore] = []
    cover_letters: Dict[str, str] = {}
