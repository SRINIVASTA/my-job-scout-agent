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

# Initialize system session access states
if "activated" not in st.session_state:
    st.session_state.activated = False

# 🔑 SIDEBAR INPUT AGGREGATOR PANEL
with st.sidebar:
    st.markdown("### 🔑 Secure Dashboard Access")
    
    # Check if keys are present in the Streamlit secrets structure
    if "MASTER_PASSWORD" not in st.secrets:
        st.error("🔒 Configuration Error: 'MASTER_PASSWORD' not discovered inside secrets.toml file space.")
        st.stop()
        
    user_password_input = st.text_input(
        "Enter Master Password", 
        type="password", 
        help="Input your private security password to unlock background API execution frameworks."
    )
    
    if user_password_input:
        if user_password_input == st.secrets["MASTER_PASSWORD"]:
            st.session_state.activated = True
            st.success("🔓 Dashboard Unlocked Successfully!")
        else:
            st.session_state.activated = False
            st.sidebar.error("❌ Invalid Password Token. Access Denied.")
            
    st.markdown("---")
    st.markdown("### ⚙️ Workflow Configuration")
    
    selected_time_window = st.selectbox(
        "Scrape Live Postings Within:",
        options=["Within 1 Hour", "Past 24 Hours", "Past 3 Days"],
        index=2,
        disabled=not st.session_state.activated,
        help="Filters live Firecrawl web crawl targets to match your exact application timeline."
    )
    
    match_threshold = st.slider(
        "Minimum Fit Score Threshold (%)",
        min_value=0, max_value=100, value=50, step=5,
        disabled=not st.session_state.activated,
        help="Roles scoring below this percentage will be flagged with a FIT GAP DETECTED alert."
    )
    st.markdown("---")
    st.caption("Secure Vault Enabled Engine 🔒")

# 🖥️ MAIN GATEKEEPER DISPLAY LAYER
st.title("🤖 Live LangGraph & Gemini Job Matching Agent")

if not st.session_state.activated:
    st.warning("🔒 Access Restricted: Please provide the valid custom Master Password inside the sidebar configuration box to unlock your processing agents.")
    st.stop()

st.markdown("Tracking opportunities with threshold-highlighted dashboards and professional tailored report extractions.")

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
        # Fetch tokens implicitly straight out of secret configuration spaces
        google_api_key = st.secrets.get("GOOGLE_API_KEY", "")
        hf_token = st.secrets.get("HF_TOKEN", "")
        firecrawl_key = st.secrets.get("FIRECRAWL_API_KEY", "")
        
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            
        with st.spinner(f"Orchestrating graph nodes with Firecrawl query filtered to {selected_time_window}..."):
            try:
                initial_state = {
                    "cv_text": raw_cv_text,
                    "match_threshold": match_threshold,
                    "time_filter": selected_time_window.lower(),
                    "google_api_key": google_api_key,
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
                                        clean_title = job_title.replace(" ", "_").lower()
                                        
                                        st.download_button(
                                            label="📥 Download Cover Letter (.txt)",
                                            data=target_letter,
                                            file_name=f"cover_letter_{clean_title}.txt",
                                            mime="text/plain",
                                            key=f"txt_dl_{index}"
                                        )
                                        
                                        pdf_data = generate_pdf_document(candidate_name, job_title, target_letter)
                                        st.download_button(
                                            label="📥 Download Cover Letter (PDF Only)",
                                            data=pdf_data,
                                            file_name=f"cover_letter_{clean_title}.pdf",
                                            mime="application/pdf",
                                            key=f"pdf_dl_{index}"
                                        )
                                else:
                                    st.caption("Cover letter generated exclusively for roles surging past parameters.")
                                    
            except Exception as pipeline_error:
                st.error(f"An unexpected disruption occurred during LangGraph node orchestration: {pipeline_error}")
                st.exception(pipeline_error)
