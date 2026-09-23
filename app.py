import streamlit as st
import plotly.graph_objects as go
import pypdf
import os
import io
from graph_engine import agent_app

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

st.set_page_config(page_title="Gemini AI Job Scout", page_icon="🤖", layout="wide")

st.title("🤖 Live LangGraph & Gemini Job Matching Agent")
st.caption("Tracking live opportunities with threshold-highlighted dashboards and professional PDF report exports.")

def generate_report_pdf(final_output):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=22, spaceAfter=15, textColor="#1E3A8A")
    h2_style = ParagraphStyle('H2Style', parent=styles['Heading2'], fontSize=14, spaceBefore=12, spaceAfter=6, textColor="#10B981")
    body_style = ParagraphStyle('BodyStyle', parent=styles['BodyText'], fontSize=10, leading=14, spaceAfter=8)
    
    story.append(Paragraph("AI Recruitment Analysis Report", title_style))
    story.append(Paragraph(f"<b>Candidate Evaluated:</b> {final_output['profile'].name}", body_style))
    story.append(Paragraph(f"<b>Profile Brief:</b> {final_output['profile'].experience_summary}", body_style))
    story.append(Paragraph(f"<b>Skills Verified:</b> {', '.join(final_output['profile'].skills)}", body_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("Market Placement Scoring", h2_style))
    for evaluation in final_output['ranked_jobs']:
        status_text = "PASSING MATCH (>=70%)" if evaluation.threshold_passed else "FIT GAP IDENTIFIED"
        story.append(Paragraph(f"• <b>{evaluation.job_title}</b>: Match Score: {evaluation.fit_score}/100 [{status_text}]", body_style))
        story.append(Paragraph(f"<i>Gap Analysis:</i> {evaluation.gap_explanation}", body_style))
        story.append(Spacer(1, 10))
        
    if final_output['cover_letters']:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Auto-Drafted Cover Letters (Threshold Passed Positions)", h2_style))
        for j_id, letter_text in final_output['cover_letters'].items():
            job_obj = next((j for j in final_output['ranked_jobs'] if j.job_id == j_id), None)
            j_title = job_obj.job_title if job_obj else "Target Position"
            story.append(Paragraph(f"<b>Application Pack for: {j_title}</b>", body_style))
            formatted_letter = letter_text.replace("\n", "<br/>")
            story.append(Paragraph(formatted_letter, body_style))
            story.append(Spacer(1, 15))
            
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

with st.sidebar:
    st.header("🔑 Authentication Setup")
    api_key_input = st.text_input("Google API Key", type="password", help="Enter your Gemini API Key from Google AI Studio")
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input

uploaded_file = st.file_uploader("Drop your CV Resume here", type=["pdf"])

if uploaded_file and not api_key_input:
    st.warning("⚠️ Please input your Google API Key into the sidebar to authorize the agent workflows.")

elif uploaded_file and api_key_input:
    if st.button("🚀 Run Live Match Engine Pipeline"):
        with st.spinner("Streaming live web metrics and running analytical graphs..."):
            pdf_reader = pypdf.PdfReader(uploaded_file)
            extracted_cv_text = ""
            for page in pdf_reader.pages:
                text_content = page.extract_text()
                if text_content:
                    extracted_cv_text += text_content + "\n"
            
            initial_state = {"cv_text": extracted_cv_text}
            final_output = agent_app.invoke(initial_state)
            st.session_state["pipeline_results"] = final_output
            st.success("🎉 Pipeline Complete!")
            
    if "pipeline_results" in st.session_state:
        final_output = st.session_state["pipeline_results"]
        pdf_bytes = generate_report_pdf(final_output)
        st.download_button(
            label="📥 Download Complete Analysis & Application Pack (.PDF)",
            data=pdf_bytes,
            file_name="AI_Job_Scout_Report.pdf",
            mime="application/pdf"
        )
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Candidate Profile Extraction Matrix")
            st.write(f"**Name:** {final_output['profile'].name}")
            st.write(f"**Background Profile:** {final_output['profile'].experience_summary}")
            st.write("**Extracted Skill Badges:**")
            st.write(", ".join([f"`{skill}`" for skill in final_output['profile'].skills]))
            st.info(f"Targeting vacancies using query string: **'{final_output['search_query']}'**")
        
        with col2:
            st.subheader("📊 Threshold-Highlighted Placement Metrics")
            job_titles = [item.job_title for item in final_output['ranked_jobs']]
            scores = [item.fit_score for item in final_output['ranked_jobs']]
            
            THRESHOLD = 70
            bar_colors = ["#10B981" if s >= THRESHOLD else "#EF4444" for s in scores]
            text_positions = ["Matched ✅" if s >= THRESHOLD else "Gap Identified ❌" for s in scores]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=scores, y=job_titles, orientation='h', marker_color=bar_colors,
                text=[f"  {s}% - {pos}" for s, pos in zip(scores, text_positions)],
                textposition='outside', hovertemplate="<b>Match Score:</b> %{x}%<extra></extra>"
            ))
            fig.add_shape(
                type="line", x0=THRESHOLD, y0=-0.5, x1=THRESHOLD, y1=len(job_titles)-0.5,
                line=dict(color="Gold", width=3, dash="dashdot")
            )
            fig.update_layout(
                xaxis=dict(title="Match Suitability Percentage (%)", range=[0, 110]),
                yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=30, b=20), height=350,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        st.divider()
        st.subheader("🔎 In-Depth Job Positioning Reports")
        for evaluation in final_output['ranked_jobs']:
            status_badge = "🟢 PASSING MATCH" if evaluation.threshold_passed else "🔴 FIT GAP DETECTED"
            with st.expander(f"📍 {evaluation.job_title} — Score: {evaluation.fit_score}/100 [{status_badge}]"):
                tab1, tab2 = st.tabs(["Skill Gap Critique", "Tailored Cover Letter 📝"])
                with tab1:
                    st.markdown("**Missing Technical Attributes / Skill Gap Critique:**")
                    st.write(evaluation.gap_explanation)
                with tab2:
                    if evaluation.threshold_passed:
                        letter_data = final_output['cover_letters'].get(evaluation.job_id, "Generating letter structure...")
                        st.text_area("Auto-Generated Cover Letter Draft", value=letter_data, height=300, key=evaluation.job_id)
                    else:
                        st.info("💡 Cover letters are automatically skipped for applications that sit under the 70% suitability threshold.")
