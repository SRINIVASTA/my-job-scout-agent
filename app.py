import os
import streamlit as st
from pypdf import PdfReader
from graph_engine import agent_app
from nodes import generate_pdf_document

st.set_page_config(
    page_title="Gemini & LangGraph Job Scout Agent",
    page_icon="🤖",
    layout="wide"
)

# 🔑 SIDEBAR INPUT AGGREGATOR PANEL
with st.sidebar:
    st.markdown("### 🔑 Authentication Setup")
    google_key = st.text_input("Google API Key", type="password", help="Primary LLM processor")
    hf_token = st.text_input("Hugging Face Token", type="password", help="Automatic fallback LLM runner token")
    firecrawl_key = st.text_input("Firecrawl API Key", type="password", help="Live web scraping engine key")
    
    st.markdown("---")
    st.markdown("### ⚙️ Workflow Configuration")
    match_threshold = st.slider(
        "Minimum Fit Score Threshold (%)",
        min_value=0, max_value=100, value=70, step=5,
        help="Roles scoring below this percentage will be flagged with a FIT GAP DETECTED alert."
    )
    st.markdown("---")
    st.caption("Robust Fallback Enabled Engine 🦜🔗")

st.title("🤖 Live LangGraph & Gemini Job Matching Agent")
st.markdown("Tracking live opportunities with threshold-highlighted dashboards and professional tailored report extractions.")

# Verify that at least one functional text-processing token configuration exists
if not google_key and not hf_token:
    st.warning("Please provide either a Google API Key or a Hugging Face Token in the sidebar to activate the processing agents.")
    st.stop()

# 📥 RESUME UPLOAD UTILITY
uploaded_file = st.file_uploader("Drop your CV Resume here", type=["pdf"])

if uploaded_file is not None:
    try:
        pdf_reader = PdfReader(uploaded_file)
        raw_cv_text = ""
        for page in pdf_reader.pages:
            text_content = page.extract_text()
            if text_content: 
                raw_cv_text += text_content + "\n"
        if not raw_cv_text.strip():
            st.error("Could not extract legible text from your PDF. Please ensure it is not a flat image scan.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading PDF file structure: {e}")
        st.stop()

    # 🚀 ORCHESTRATION PIPELINE BUTTON
    if st.button("🚀 Run Agent Pipeline Framework", use_container_width=True):
        if google_key:
            os.environ["GOOGLE_API_KEY"] = google_key
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            
        with st.spinner("Orchestrating multi-agent graph nodes (Extraction -> Firecrawl Search -> Ranking -> Generation)..."):
            try:
                initial_state = {
                    "cv_text": raw_cv_text,
                    "match_threshold": match_threshold,
                    "google_api_key": google_key,
                    "hf_token": hf_token,
                    "firecrawl_api_key": firecrawl_key,
                    "raw_jobs": [],
                    "ranked_jobs": [],
                    "cover_letters": {}
                }
                
                final_output_state = agent_app.invoke(initial_state)
                st.success("🎉 Pipeline Complete!")
                st.markdown("---")
                
                # Render profile metrics layout
                profile = final_output_state.get("profile")
                if profile:
                    st.subheader("📋 Candidate Profile Extraction Matrix")
                    st.markdown(f"**Name:** {getattr(profile, 'name', 'Unknown Candidate')}")
                    st.markdown(f"**Background Profile:** {getattr(profile, 'experience_summary', 'N/A')}")
                    skills_list = getattr(profile, 'skills', [])
                    if skills_list:
                        st.markdown("**Extracted Skill Badges:**")
                        st.markdown(" ".join([f"`{s.strip()}`" for s in skills_list]))
                
                search_query = final_output_state.get("search_query", "N/A")
                st.info(f"Targeting vacancies using auto-generated agent query string: **'{search_query}'**")
                st.markdown("---")
                
                # Display job match panels
                st.subheader("📊 Threshold-Highlighted Placement Metrics")
                ranked_jobs = final_output_state.get("ranked_jobs", [])
                
                if not ranked_jobs:
                    st.warning("No job openings evaluated.")
                else:
                    st.markdown("### 🔎 In-Depth Job Positioning Reports")
                    for index, score_card in enumerate(ranked_jobs):
                        score = getattr(score_card, 'fit_score', 0)
                        color_container = st.success if score >= match_threshold else st.error
                        status_badge = f"🍏 **MATCH PASSED** ({score}/100)" if score >= match_threshold else f"🔴 **FIT GAP DETECTED** ({score}/100)"
                        
                        with color_container(f"📍 {getattr(score_card, 'job_title', 'Role Evaluation')} — {status_badge}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Critique Evaluation Matrix:**")
                                st.write(getattr(score_card, 'gap_explanation', 'No analysis details provided.'))
                            with col2:
                                st.markdown("**Generated Document Utilities:**")
                                cover_letters_map = final_output_state.get("cover_letters", {})
                                job_id = getattr(score_card, 'job_id', '')
                                target_letter = cover_letters_map.get(job_id)
                                
                                if target_letter:
                                    with st.expander("📄 View Tailored Cover Letter Content"):
                                        st.text_area(label="Raw Output Copy", value=target_letter, height=250, key=f"txt_{index}")
                                        
                                        candidate_name = getattr(profile, 'name', 'Candidate')
                                        job_title = getattr(score_card, 'job_title', 'Target Role')
                                        pdf_data = generate_pdf_document(candidate_name, job_title, target_letter)
                                        clean_title = job_title.replace(" ", "_").lower()
                                        
                                        # 📥 EXCLUSIVE PDF COMPILATION EXPORT BUTTON
                                        st.download_button(
                                            label="📥 Download Cover Letter (PDF Only)",
                                            data=pdf_data,
                                            file_name=f"cover_letter_{clean_title}.pdf",
                                            mime="application/pdf",
                                            key=f"pdf_dl_{index}"
                                        )
                                else:
                                    st.caption("Cover letter generated exclusively for roles surpassing target configuration threshold settings.")
                                    
            except Exception as pipeline_error:
                st.error(f"An unexpected disruption occurred during LangGraph node orchestration: {pipeline_error}")
                st.exception(pipeline_error)
