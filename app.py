import streamlit as st
import os
from pypdf import PdfReader
# Keep your existing imports, schemas, and LangGraph pipeline compilation intact above this point...

# --- 6. STREAMLIT FRONTEND DASHBOARD WITH FILE ATTACHMENTS ---
st.set_page_config(page_title="AI Job Scout", layout="wide", page_icon="🤖")
st.title("🤖 LangGraph AI Job Scout Agent")
st.caption("Powered by Google Gemini 2.5 & LangGraph")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Input Profile")
    
    # Toggle between Uploading a File vs Custom Text Input
    input_method = st.radio("Choose how to provide your CV:", ["Upload PDF File", "Copy-Paste Text Input"])
    
    cv_text_to_process = ""
    
    if input_method == "Upload PDF File":
        uploaded_file = st.file_uploader("Drag and drop your CV profile here (PDF format):", type=["pdf"])
        if uploaded_file is not None:
            with st.spinner("📄 Reading PDF text data..."):
                try:
                    # Initialize PyPDF Reader on the uploaded binary stream
                    reader = PdfReader(uploaded_file)
                    extracted_text = ""
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            extracted_text += text + "\n"
                    
                    if extracted_text.strip():
                        cv_text_to_process = extracted_text
                        st.success(f"✅ Successfully extracted text from '{uploaded_file.name}'!")
                        # Preview snippet of the document to the user
                        with st.checkbox("🔍 Preview Extracted Text"):
                            st.text_area("Extracted Payload Preview", value=cv_text_to_process[:800] + "...", height=150, disabled=True)
                    else:
                        st.error("⚠️ Could not read text layers from this PDF. It might be scanned as a flat image.")
                except Exception as e:
                    st.error(f"❌ Failed to parse document: {e}")
                    
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
