import os
import streamlit as st
from pypdf import PdfReader
# Import your LangGraph application instance from your engine file
from graph_engine import agent_app

# ⚙️ STREAMLIT PAGE SETUP
st.set_page_config(
    page_title="Gemini & LangGraph Job Scout Agent",
    page_icon="🤖",
    layout="wide"
)

# 🔑 SIDEBAR SETUP
with st.sidebar:
    st.markdown("### 🔑 Authentication Setup")
    api_key = st.text_input("Google API Key", type="password")
    
    st.markdown("---")
    st.markdown("### ⚙️ Workflow Configuration")
    # Dynamic match threshold configuration slider
    match_threshold = st.slider(
        "Minimum Fit Score Threshold (%)",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
        help="Roles scoring below this percentage will be marked with a FIT GAP DETECTED alert."
    )
    
    st.markdown("---")
    st.caption("Powered by LangGraph 🦜🔗 & Gemini-2.5-Flash")

# 🖥️ MAIN APPLICATION INTERFACE
st.title("🤖 Live LangGraph & Gemini Job Matching Agent")
st.markdown("Tracking live opportunities with threshold-highlighted dashboards and professional tailored report extractions.")

# Check for API Key entry
if not api_key:
    st.warning("Please enter your Google API Key in the left sidebar to activate the processing agents.")
    st.stop()

# 📥 RESUME UPLOAD COMPONENT
uploaded_file = st.file_uploader("Drop your CV Resume here", type=["pdf"])

if uploaded_file is not None:
    # Read text contents from uploaded PDF file binary stream
    try:
        pdf_reader = PdfReader(uploaded_file)
        raw_cv_text = ""
        for page in pdf_reader.pages:
            text_content = page.extract_text()
            if text_content:
                raw_cv_text += text_content + "\n"
                
        if not raw_cv_text.strip():
            st.error("Could not extract legible text from your PDF. Please ensure it is not an image scan.")
            st.stop()
            
    except Exception as e:
        st.error(f"Error reading PDF file structure: {e}")
        st.stop()

    # 🚀 PIPELINE RUNTRIGGER
    if st.button("🚀 Run Agent Pipeline Framework", use_container_width=True):
        # ENV VARIABLE BINDING FIX: Injects key context globally into active environment runtime space
        os.environ["GOOGLE_API_KEY"] = api_key
        
        with st.spinner("Orchestrating multi-agent graph nodes (Extraction -> Search -> Ranking -> Generation)..."):
            try:
                # Construct initial compilation state dictionary, including our new UI slider value
                initial_state = {
                    "cv_text": raw_cv_text,
                    "match_threshold": match_threshold,  # Pass threshold state downstream to graph nodes
                    "raw_jobs": [],
                    "ranked_jobs": [],
                    "cover_letters": {}
                }
                
                # Execute LangGraph runtime pipeline synchronously
                final_output_state = agent_app.invoke(initial_state)
                
                st.success("🎉 Pipeline Complete!")
                st.markdown("---")
                
                # 📋 RENDERING NODE 1 OUTPUT: CANDIDATE PROFILE EXTRACTION MATRIX
                profile = final_output_state.get("profile")
                if profile:
                    st.subheader("📋 Candidate Profile Extraction Matrix")
                    st.markdown(f"**Name:** {getattr(profile, 'name', 'Unknown Candidate')}")
                    st.markdown(f"**Background Profile:** {getattr(profile, 'experience_summary', 'N/A')}")
                    
                    # Style skills as beautiful markdown badges
                    skills_list = getattr(profile, 'skills', [])
                    if skills_list:
                        st.markdown("**Extracted Skill Badges:**")
                        badges = " ".join([f"`{skill.strip()}`" for skill in skills_list])
                        st.markdown(badges)
                
                search_query = final_output_state.get("search_query", "N/A")
                st.info(f"Targeting vacancies using auto-generated agent query string: **'{search_query}'**")
                st.markdown("---")
                
                # 📊 RENDERING NODE 3 OUTPUT: THRESHOLD HIGHLIGHTED METRICS
                st.subheader("📊 Threshold-Highlighted Placement Metrics")
                ranked_jobs = final_output_state.get("ranked_jobs", [])
                
                if not ranked_jobs:
                    st.warning("No job openings evaluated or fallback nodes triggered.")
                else:
                    st.markdown("### 🔎 In-Depth Job Positioning Reports")
                    
                    for index, score_card in enumerate(ranked_jobs):
                        # Determine badge layout status contextually using the current threshold slider bounds
                        score = getattr(score_card, 'fit_score', 0)
                        
                        # Dynamically check threshold bounds again on UI layout render step
                        if score >= match_threshold:
                            status_badge = f"🍏 **MATCH PASSED** ({score}/100)"
                            color_container = st.success
                        else:
                            status_badge = f"🔴 **FIT GAP DETECTED** ({score}/100)"
                            color_container = st.error
                        
                        # Draw high-visibility layout boxes per evaluation entry
                        with color_container(f"📍 {getattr(score_card, 'job_title', 'Role Evaluation')} — {status_badge}"):
                            # Split into side-by-side informational columns using proper width definitions
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Critique Evaluation Matrix:**")
                                # PROPERTY RESOLUTION FIX: Swapped out missing 'critique' attribute for real schema 'gap_explanation' field
                                st.write(getattr(score_card, 'gap_explanation', 'No analysis details provided by model node.'))
                            
                            with col2:
                                st.markdown("**Generated Document Utilities:**")
                                # Read cover letter text map from final state matching this exact job ID
                                cover_letters_map = final_output_state.get("cover_letters", {})
                                job_id = getattr(score_card, 'job_id', '')
                                target_letter = cover_letters_map.get(job_id)
                                
                                if target_letter:
                                    # Show expandable text box containing tailored copy
                                    with st.expander("📄 View Tailored Cover Letter Content"):
                                        st.text_area(label="Raw Output Copy", value=target_letter, height=300, key=f"txt_{index}")
                                        
                                        # 📥 UTILITY UPGRADE: Instant one-click file download option for the user
                                        clean_title = getattr(score_card, 'job_title', 'role').replace(" ", "_").lower()
                                        st.download_button(
                                            label="📥 Download Cover Letter (.txt)",
                                            data=target_letter,
                                            file_name=f"cover_letter_{clean_title}.txt",
                                            mime="text/plain",
                                            key=f"dl_{index}"
                                        )
                                else:
                                    st.caption("Cover letter generated exclusively for roles surpassing target pipeline configuration threshold settings.")
                                    
            except Exception as pipeline_error:
                st.error(f"An unexpected disruption occurred during LangGraph node orchestration tracking: {pipeline_error}")
                st.exception(pipeline_error)
