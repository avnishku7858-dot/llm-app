# LLM APIs & Application Development: Complete Milestone Solution

This repository contains the complete, production-grade implementation of the **LLM APIs & Application Development** curriculum module, including all four required objective deliverables and the unlocked milestone reward.

---

## 📋 Course Objectives & Architecture Map

| Objective | Description | Deliverable File | Key Technologies |
|---|---|---|---|
| **1** | Understand LLM APIs, authentication, environment variables & connect LLMs | [`01_authentication_and_setup.py`](file:///C:/Users/ASUS/.gemini/antigravity/scratch/llm-apis-and-apps/01_authentication_and_setup.py) | `google-genai`, `dotenv`, `os.environ` |
| **2** | Manage conversations, chat history, system/user/assistant messages & optimize parameters/tokens | [`02_conversation_and_token_management.py`](file:///C:/Users/ASUS/.gemini/antigravity/scratch/llm-apis-and-apps/02_conversation_and_token_management.py) | Sliding Window Context, Temperature, Top-P, Max Tokens |
| **3** | Implement structured responses, robust error handling & basic API cost management | [`03_structured_outputs_and_cost.py`](file:///C:/Users/ASUS/.gemini/antigravity/scratch/llm-apis-and-apps/03_structured_outputs_and_cost.py) | `Pydantic`, Exponential Backoff, Cost formulas |
| **4** | Build the AI Chat Application featuring user input, AI response, history, custom instructions & UI | [`04_ai_chat_app.py`](file:///C:/Users/ASUS/.gemini/antigravity/scratch/llm-apis-and-apps/04_ai_chat_app.py) | `Streamlit`, Real-time Streaming, Token/Cost Tracker |
| **Reward**| Unlocked Milestone Reward: High-impact ATS Resume Templates | [`resume_templates/`](file:///C:/Users/ASUS/.gemini/antigravity/scratch/llm-apis-and-apps/resume_templates/) | ATS formatting guide & templates |

---

## 🚀 Quick Start Guide

### 1. Clone / Navigate to Project Directory
```powershell
cd "C:\Users\ASUS\.gemini\antigravity\scratch\llm-apis-and-apps"
```

### 2. Activate Virtual Environment
A dedicated virtual environment has already been created and populated:
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### 3. Configure API Credentials
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` and set your API key:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key
DEFAULT_MODEL=gemini-3.8-flash
DEFAULT_TEMPERATURE=0.7
```
*(Note: All scripts and the application include simulation fallbacks, meaning they run and demonstrate every concept even before adding a live key!)*

---

## 🧪 Testing Each Milestone Objective

### Objective 1: Authentication & LLM Connectivity
```powershell
python 01_authentication_and_setup.py
```
- Validates environment variable loading via `python-dotenv`.
- Performs masked credential verification without leaking secrets.
- Executes an API connection test with error trapping.

### Objective 2: Conversation Management & Token Optimization
```powershell
python 02_conversation_and_token_management.py
```
- Demonstrates multi-turn state persistence.
- Enforces system instruction personas across multiple user turns.
- Executes **Sliding Window Pruning** to keep context size constrained.
- Tracks input vs output token consumption.

### Objective 3: Structured Outputs, Robust Error Handling & Cost Tracking
```powershell
python 03_structured_outputs_and_cost.py
```
- Uses **Pydantic** (`ATSResumeEvaluation`) for deterministic JSON output schemas.
- Implements exponential backoff with jitter for HTTP 429 rate limit resilience.
- Calculates micro-dollar API costs ($/1M tokens) with threshold budget warnings.

### Objective 4: Launching the AI Chat Application
Launch the interactive Streamlit UI:
```powershell
streamlit run 04_ai_chat_app.py
```
**UI Features:**
- **Custom System Instructions**: Choose from presets (Technical Mentor, ATS Resume Coach, Socratic Tutor) or write custom instructions.
- **Model & Hyperparameters**: Real-time sliders for Temperature, Top-P, Max Output Tokens, and Sliding Window size.
- **Real-Time Streaming**: Live streaming token generation.
- **Session Metrics**: Live prompt token counter, completion token counter, and session USD expenditure.
- **Export & Reset**: One-click Chat History JSON export and session purge.

---

## 🎁 Milestone Reward: ATS Resume Templates

Located in the [`resume_templates/`](file:///C:/Users/ASUS/.gemini/antigravity/scratch/llm-apis-and-apps/resume_templates/) directory:
1. **[ATS AI & ML Engineer Template](file:///C:/Users/ASUS/.gemini/antigravity/scratch/llm-apis-and-apps/resume_templates/ATS_AI_ML_Engineer_Template.md)**: Highlights LLM pipelines, token optimization, prompt engineering, and cloud deployment.
2. **[ATS Software Engineer Template](file:///C:/Users/ASUS/.gemini/antigravity/scratch/llm-apis-and-apps/resume_templates/ATS_Software_Engineer_Template.md)**: Focuses on backend microservices, REST APIs, scalability metrics, and database performance.
3. **[ATS Optimization Guide](file:///C:/Users/ASUS/.gemini/antigravity/scratch/llm-apis-and-apps/resume_templates/ATS_Optimization_Guide.md)**: Proven Google XYZ bullet formula, parsing rules for Greenhouse/Workday/Lever, and high-converting keyword checklist.
