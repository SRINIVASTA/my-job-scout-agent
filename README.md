# 🤖 Live LangGraph & Gemini Job Matching Agent with Robust Fallback

An advanced, enterprise-grade AI recruiting agent workflow built using **LangGraph**, **Streamlit**, and **LangChain**. The application securely ingests a candidate's resume (PDF), extracts profile parameters, dynamically searches for active roles matching time-horizons (up to within 1 hour), calculates placement compatibility scores, and writes complete, un-truncated cover letters available in both plain text and **ReportLab PDF** binaries.

The engine features a **fail-safe architectural fallback framework**: if the primary **Google Gemini 2.5 Flash** endpoint experiences rate limits or token exhaustion (`429 Resource Exhausted`), the workflow automatically routes conversational workloads to **Hugging Face's serverless infrastructure** using `Qwen/Qwen2.5-Coder-7B-Instruct` chat message contexts.

---

## 🔒 Security & Secrets Vault Management
To comply with strict production code separation standards, **no hardcoded API keys exist inside the source code**. All credentials and target tokens are locked away in a background configurations file (`.streamlit/secrets.toml`). 

The application is completely locked behind a **Gatekeeper Master Password** field in the UI sidebar. The processing backend unlocks *only* when the user supplies the correct private token matching your environment profile.

---

## 📂 Repository File Tree Structure

Ensure your workspace directory tree matches this exact layout framework:

```text
my-job-scout-agent/
│
├── app.py                  # 🖥️ Interactive Streamlit Interface Dashboard UI
├── schemas.py              # 📊 Pydantic State & Shared Memory Models
├── nodes.py                # ⚙️ Core Workflow Actions & ReportLab PDF Engine 
├── graph_engine.py         # 🦜 LangGraph StateGraph Execution Pipeline
└── requirements.txt        # 📦 Dependency version manifest records
```

---

## ⚙️ Setup & Deployment Guide

### 1. Clone or Create your Local Environment Workspace
Initialize a directory folder and navigate into it:
```bash
mkdir my-job-scout-agent
cd my-job-scout-agent
```

### 2. Install All System Dependencies
Make sure you have Python 3.10+ installed. Build your environment dependencies by running:
```bash
pip install -r requirements.txt
```

### 3. Boot Up the Dashboard Engine
Execute the Streamlit application module directly inside your terminal root:
```bash
streamlit run app.py
```

---

## 🕹️ Deep-Dive Workflow Pipeline Execution

1. **Dashboard Authentication:** The screen initiates under an **"Access Restricted"** state lock. Type your configured `MASTER_PASSWORD` into the sidebar field to open up the environment variables.
2. **Timeline Filters Selection:** Toggle between options to restrict Firecrawl target queries contextually (e.g., **"Within 1 Hour"**, **"Past 24 Hours"**, or **"Past 3 Days"**).
3. **Threshold Calibration Slider:** Tune your score threshold bar. Lowering it down to **50%** ensures that competitive matching matrices execute document nodes gracefully.
4. **Resume Injection:** Drop your `Resume.pdf` file into the upload box wrapper stream.
5. **Run Agent Pipeline Core:** Hit the execution button to trigger `agent_app.invoke(initial_state)`. Watch the LangGraph state orchestration framework extract parameters, route queries, score profiles, and return full un-cut output strings alongside instant **PDF download actions** side-by-side!

---

## 🦜 Tech Stack Framework Highlights
* **Orchestration:** LangGraph (StateGraph compilation mapping)
* **LLM Engine:** Gemini-2.5-Flash (Primary) 🔄 Qwen-2.5-Coder-7B-Instruct (Fail-safe Chat Fallback)
* **Scraping Engine:** Firecrawl Search Web-Scale parameters configuration
* **Binary Compilation:** ReportLab Flowables Page Blueprint
* **Frontend Canvas:** Streamlit Data Ecosystem
