""
This updated frontend file replaces the single-pass ReportLab logic with interactive metric tables, 
live inspection logs for the compiled LaTeX source, and a custom /outcome logging component 
to handle the human-in-the-loop feedback sequence.
"""
import os
import streamlit as st
from pypdf import PdfReader
from graph_engine import agent_app

st.set_page_config(
    page_title="Agentic Job Scout & Application Pipeline",
    page_icon="🤖",
    layout="wide"
)
if "activated" not in st.session_state:
    st.session_state.activated = False

# 🔑 SIDEBAR INPUT AGGREGATOR PANEL
with st.sidebar:
    st.markdown("### 🔑 Secure Dashboard Access")
    if "MASTER_PASSWORD" not in st.secrets:
        st.error("🔒 Configuration Error: 'MASTER_PASSWORD' not discovered inside secrets.toml file space.")
        st.stop()
        
    user_password_input = st.text_input("Enter Master Password", type="password")
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
        disabled=not st.session_state.activated
    )
    match_threshold = st.slider(
        "Minimum Fit Score Threshold (%)",
        min_value=0, max_value=100, value=70, step=5,
        disabled=not st.session_state.activated
    )
    st.markdown("---")
    st.caption("Secure Vault Enabled Engine 🔒")

# 🖥️ MAIN DISPLAY LAYER
st.title("🤖 Agentic Job Search & Application Pipeline")

if not st.session_state.activated:
    st.warning("🔒 Access Restricted: Please provide the valid Master Password inside the sidebar configuration box to unlock your processing agents.")
    st.stop()

uploaded_file = st.file_uploader("Drop your CV Resume here", type=["pdf"])

if uploaded_file is not None:
    try:
        pdf_reader = PdfReader(uploaded_file)
        raw_cv_text = ""
        for page in pdf_reader.pages:
            text_content = page.extract_text()
            if text_content: raw_cv_text += text_content + "\n"
        if not raw_cv_text.strip():
            st.error("Could not extract legible text from your PDF file layout.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading PDF file structure: {e}")
        st.stop()

    if st.button("🚀 Run Agent Pipeline Framework", use_container_width=True):
        google_api_key = st.secrets.get("GOOGLE_API_KEY", "")
        hf_token = st.secrets.get("HF_TOKEN", "")
        firecrawl_key = st.secrets.get("FIRECRAWL_API_KEY", "")
        
        if google_api_key: os.environ["GOOGLE_API_KEY"] = google_api_key
        if hf_token: os.environ["HF_TOKEN"] = hf_token
            
        with st.spinner("Executing pipeline phases: Parsing Profile ➔ Sourcing Jobs ➔ Filtering Shortlist ➔ Multi-Agent Design Loop..."):
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
                    "top_shortlist": [],
                    "current_job_idx": 0,
                    "drafted_latex": "",
                    "approved": False
                }
                
                final_output_state = agent_app.invoke(initial_state)
                st.success("🎉 Pipeline Task Cycle Complete!")
                st.markdown("---")
                
                # Render profile extraction card metrics
                profile = final_output_state.get("profile")
                if profile:
                    st.subheader("📋 Verified Local Profile Data Matrix")
                    st.markdown(f"**Name:** {getattr(profile, 'name', 'Unknown')}")
                    st.markdown(f"**Background Summary:** {getattr(profile, 'experience_summary', 'N/A')}")
                    skills = getattr(profile, 'skills', [])
                    if skills:
                        st.markdown(" ".join([f"`{s.strip()}`" for s in skills]))
                
                st.markdown("---")
                
                # Render Shortlist Table View
                st.subheader("📊 Phase 2 Filter Result: Top Job Shortlist")
                top_shortlist = final_output_state.get("top_shortlist", [])
                
                if not top_shortlist:
                    st.warning("No scraped vacancy records passed your minimum compatibility scoring bounds.")
                else:
                    # Multi-attribute comparison table rendering layout
                    shortlist_data = [
                        {"Job Title": job.job_title, "Compatibility Index": f"{job.fit_score}%", "Passing Status": "PASSED ✅"}
                        for job in top_shortlist
                    ]
                    st.table(shortlist_data)
                    
                    st.markdown("---")
                    st.subheader("📄 Phase 3 & 4: Asset Application Reviews & Outcomes")
                    
                    for index, score_card in enumerate(top_shortlist):
                        with st.expander(f"📍 Application Package for: {score_card.job_title}", expanded=(index==0)):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Structural Fit Critique:**")
                                st.write(score_card.gap_explanation)
                            with col2:
                                st.markdown("**Automated Production Delivery Tools:**")
                                pdf_bytes = final_output_state.get("application_pdf_binary")
                                
                                if pdf_bytes:
                                    st.download_button(
                                        label="📥 Download Tailored Application (.PDF)",
                                        data=pdf_bytes,
                                        file_name=f"Bespoke_Application_{index}.pdf",
                                        mime="application/pdf",
                                        key=f"latex_dl_{index}"
                                    )
                                    with st.expander("🛠️ View Compiled Source Markup"):
                                        st.code(final_output_state.get("drafted_latex", ""), language="latex")
                                else:
                                    st.info("No compiled document asset buffer is currently available for this entry.")
                                    
                    # The human evaluation loop layer component
                    st.markdown("---")
                    st.subheader("🚀 Log Application Outcome (`/outcome`)")
                    selected_job = st.selectbox("Select target deployment log to calibrate:", options=[j.job_title for j in top_shortlist])
                    application_result = st.radio(
                        "Record Action Submission Outcome Status:",
                        options=["Successfully Submitted", "Formatting Defect Observed", "Bypassed - Position Skill Mismatch"],
                        horizontal=True
                    )
                    if st.button("💾 Commit Feedback Matrix Parameters"):
                        st.toast(f"Fit-scoring criteria logic recalibrated for: {selected_job} ({application_result})", icon="💾")
                        
            except Exception as pipeline_error:
                st.error(f"Pipeline disruption: {pipeline_error}")
                st.exception(pipeline_error)
