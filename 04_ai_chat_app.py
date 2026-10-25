"""
Objective 4: Build the AI Chat Application featuring user input, AI response,
history, custom instructions, and a basic UI.

Framework: Streamlit
Features:
- Configurable System Instructions / Persona presets + Custom prompt
- Dynamic Model & Hyperparameter controls (Temperature, Top-P, Max Tokens)
- Multi-turn conversation history management with context sliding window
- Real-time response streaming
- Token counter and real-time cost estimator (USD)
- Export conversation (JSON / Markdown) and Clear Chat
- Graceful API key configuration (via .env or interactive sidebar input)
"""

import os
import json
import time
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()

# Page Setup
st.set_page_config(
    page_title="AI Chat & Assistant Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Preset Personas
PERSONA_PRESETS = {
    "Technical Mentor": "You are an experienced senior software engineer and mentor. Give clear, idiomatic code examples and concise explanations.",
    "ATS Resume & Career Coach": "You are a professional ATS resume strategist and hiring manager. Evaluate experiences, optimize action verbs, and suggest high-impact metrics.",
    "Concise General Assistant": "You are a direct, helpful assistant. Answer all questions accurately in under 3 sentences unless specifically asked for elaboration.",
    "Socratic Tutor": "You guide students through discovery by asking probing questions and giving conceptual hints rather than raw answers.",
    "Custom": ""
}

# Pricing per million tokens (USD)
PRICING_MAP = {
    "gemini-3.8-flash": {"in": 0.15, "out": 0.60},
    "gemini-3.5-flash-lite": {"in": 0.075, "out": 0.30},
    "gemini-4-argon": {"in": 1.25, "out": 5.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
}


def estimate_tokens(text: str) -> int:
    """Heuristic approximation: ~4 characters per token."""
    return max(1, len(text) // 4)


def calculate_cost(model: str, prompt_toks: int, out_toks: int) -> float:
    rates = PRICING_MAP.get(model, {"in": 0.15, "out": 0.60})
    return (prompt_toks / 1_000_000 * rates["in"]) + (out_toks / 1_000_000 * rates["out"])


# =====================================================================
# State Initialization
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_prompt_tokens" not in st.session_state:
    st.session_state.total_prompt_tokens = 0

if "total_completion_tokens" not in st.session_state:
    st.session_state.total_completion_tokens = 0

if "total_cost_usd" not in st.session_state:
    st.session_state.total_cost_usd = 0.0


# =====================================================================
# Sidebar: Configuration & Controls
# =====================================================================
with st.sidebar:
    st.title("⚙️ AI App Controls")

    # API Key Configuration
    st.subheader("1. Authentication")
    env_gemini_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=env_gemini_key if env_gemini_key != "your_gemini_api_key_here" else "",
        type="password",
        help="Reads from .env by default, or paste here.",
    )

    active_api_key = api_key_input.strip() or env_gemini_key
    has_valid_key = bool(active_api_key and active_api_key != "your_gemini_api_key_here")

    if has_valid_key:
        st.success("API Key Active", icon="✅")
    else:
        st.warning("No API Key detected. Running in educational simulation mode.", icon="⚠️")

    st.divider()

    # Persona / Custom Instructions
    st.subheader("2. Custom Instructions")
    selected_preset = st.selectbox("Persona Preset", list(PERSONA_PRESETS.keys()))

    if selected_preset == "Custom":
        system_instruction = st.text_area(
            "Custom System Instruction",
            value="You are a helpful, thoughtful AI assistant.",
            height=100,
        )
    else:
        system_instruction = st.text_area(
            "Active System Instruction",
            value=PERSONA_PRESETS[selected_preset],
            height=100,
        )

    st.divider()

    # Model & Hyperparameters
    st.subheader("3. Model Parameters")
    model_choice = st.selectbox(
        "Target Model",
        ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-4-argon"],
        index=0,
    )

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05, help="Randomness control.")
    with col_t2:
        top_p = st.slider("Top-P", 0.1, 1.0, 0.95, 0.05, help="Nucleus sampling limit.")

    max_tokens = st.slider("Max Output Tokens", 100, 2048, 800, 50)
    history_limit = st.slider("Context Window (Turns)", 1, 15, 6, help="Sliding window memory limit.")

    st.divider()

    # Token & Cost Metrics
    st.subheader("4. Usage & Cost Metrics")
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Prompt Tokens", st.session_state.total_prompt_tokens)
    m_col2.metric("Output Tokens", st.session_state.total_completion_tokens)
    st.metric("Estimated Cost", f"${st.session_state.total_cost_usd:.6f} USD")

    st.divider()

    # Reset & Export
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_prompt_tokens = 0
        st.session_state.total_completion_tokens = 0
        st.session_state.total_cost_usd = 0.0
        st.rerun()

    if st.session_state.messages:
        chat_export_json = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            "📥 Export Chat (JSON)",
            data=chat_export_json,
            file_name="chat_history.json",
            mime="application/json",
            use_container_width=True,
        )


# =====================================================================
# Main Chat Window
# =====================================================================
st.title("🤖 Interactive AI Chat Application")
st.caption(f"Connected to **{model_choice}** | Temperature: `{temperature}` | Context Window: `{history_limit}` turns")

# Display historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Handling
user_prompt = st.chat_input("Type your message here...")

if user_prompt:
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Context Window Pruning (Sliding Window: last N turns)
    max_history_entries = history_limit * 2
    active_history = st.session_state.messages[-max_history_entries:]

    # 3. Stream or Generate AI Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        if has_valid_key:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=active_api_key)

                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_tokens,
                )

                # Build multi-turn content
                contents = []
                for entry in active_history:
                    contents.append(
                        types.Content(
                            role="user" if entry["role"] == "user" else "model",
                            parts=[types.Part.from_text(text=entry["content"])],
                        )
                    )

                # Stream response
                response_stream = client.models.generate_content_stream(
                    model=model_choice,
                    contents=contents,
                    config=config,
                )

                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

                # Calculate tokens & cost
                p_toks = estimate_tokens(system_instruction + " " + " ".join(m["content"] for m in active_history))
                c_toks = estimate_tokens(full_response)
                cost = calculate_cost(model_choice, p_toks, c_toks)

                st.session_state.total_prompt_tokens += p_toks
                st.session_state.total_completion_tokens += c_toks
                st.session_state.total_cost_usd += cost

            except Exception as e:
                full_response = f"**API Error Encountered:** `{e}`\n\n*Check your API key, model selection, or network connection.*"
                response_placeholder.markdown(full_response)
        else:
            # Simulation fallback with simulated streaming typing effect
            simulated_lines = [
                f"**[Simulated Response - {model_choice}]**\n\n",
                f"I received your message: *\"{user_prompt}\"*\n\n",
                f"- **Active Persona:** {system_instruction[:60]}...\n",
                f"- **Temperature:** `{temperature}` | **Max Tokens:** `{max_tokens}`\n",
                f"- **Context Retained:** `{len(active_history)}` total messages in sliding window.\n\n",
                "*(To enable live LLM generation, set your `GEMINI_API_KEY` in the sidebar or `.env` file.)*"
            ]
            for part in simulated_lines:
                full_response += part
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.08)

            response_placeholder.markdown(full_response)

            p_toks = estimate_tokens(user_prompt)
            c_toks = estimate_tokens(full_response)
            cost = calculate_cost(model_choice, p_toks, c_toks)

            st.session_state.total_prompt_tokens += p_toks
            st.session_state.total_completion_tokens += c_toks
            st.session_state.total_cost_usd += cost

    # Save Assistant Response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
