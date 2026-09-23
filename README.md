# 🤖 LangGraph AI Job Scout Agent

An intelligent, multi-agent automated recruiter system built using **LangGraph**, **Streamlit**, and **Google Gemini 2.5**. This application reads an uploaded resume (PDF) or raw text input, structures the profile matrix, dynamically builds target queries, simulates live competitive tech positions, and outputs analytical fit scoring with real-time **Plotly charts**.

---

## 🚀 Key Architectural Features
* **Document Extraction:** Parses raw text payloads directly from uploaded PDF CVs natively via `pypdf`.
* **Structured JSON Extraction:** Implements Pydantic schema controls (`with_structured_output`) to guarantee strict data structures from Gemini 2.5.
* **Rate-Limit Resilient Architecture:** Implements consolidated multi-job batch analysis and staggered task timing to run reliably on the **completely free Google AI Studio tier**.
* **Visual Data Analytics:** Uses an interactive **Plotly horizontal bar chart** dashboard for quick match comparison.
* **Production Security:** Features a dynamic UI Sidebar password prompt to keep API keys secure on public servers.

---

## 📁 Repository Structure
```text
my-job-scout-agent/
│
├── app.py                 # Main Streamlit dashboard script and LangGraph pipeline
├── requirements.txt       # Dependencies to support the agent
└── README.md              # Documentation setup instructions
```

---

## 🛠️ Local Installation & Setup

Follow these quick steps to launch the app locally on your desktop machine.

### 1. Clone the Code Repository
```bash
git clone https://github.com
cd my-job-scout-agent
```

### 2. Configure Your Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install All Project Dependencies
```bash
pip install -r requirements.txt
```

Your `requirements.txt` includes:
```text
streamlit
langchain-google-genai
langgraph
pydantic
pypdf
plotly
```

### 4. Execute the Local Application
```bash
streamlit run app.py
```

Once running, your terminal will print a link (usually `http://localhost:8501`). Open it, insert your free **Google Gemini API Key** into the sidebar password field, upload your CV, and trigger the workflow!

---

## 🌐 Deploying to Streamlit Community Cloud

You can take this agent live on the web as a portfolio highlight piece for free:

1. Commit and push your code files (`app.py`, `requirements.txt`) up to a public GitHub repository.
2. Navigate to [share.streamlit.io](https://my-job-scout-agent-fjidmbdgmslxrmkarrptxe.streamlit.app/) and log in using your GitHub account.
3. Click **New App**, choose your repository and the `app.py` path, then hit **Deploy**.
4. Since the key is managed directly on the app's sidebar interface, visitors can bring their own free keys from Google AI Studio to run the pipeline without using your personal API text credits.

---

## 🧠 Behind the Scenes: The LangGraph Workflow Loop

The backend engine flows sequentially across 4 state-machine graph nodes:
1. `extract_cv`: Normalizes messy text documents into a unified, typed JSON skills structure.
2. `generate_queries`: Synthesizes target keyword query combinations.
3. `fetch_jobs`: Generates realistic engineering target opportunities to match against.
4. `rank_jobs`: Evaluates candidate scores, highlights tech stack matching alignment, and uncovers target skill gaps.

---
*Developed as a practical AI Application project utilizing Agentic Workflows.*
