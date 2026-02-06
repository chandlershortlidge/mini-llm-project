import streamlit as st
import openai
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Config: change model, parameters, and options here ──
MODEL = "gpt-4o-search-preview"
MODEL_KWARGS = {
    "web_search_options": {},
    # "temperature": 0.7,       # uncomment and adjust as needed
    # "max_tokens": 4096,       # uncomment to cap response length
}

# Load system_prompt from the notebook (single source of truth)
NOTEBOOK_PATH = Path(__file__).parent / "main.ipynb"
_nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
_prompt_cell = [c for c in _nb["cells"]
                if c["cell_type"] == "code"
                and any("system_prompt" in line for line in c["source"])][0]
_ns = {}
exec("".join(_prompt_cell["source"]), _ns)
SYSTEM_PROMPT = _ns["system_prompt"]

# Prepend instructions — workaround for gpt-4o-search-preview ignoring system/developer prompts
INSTRUCTIONS = """Before giving any stock recommendations, you MUST first ask these clarifying questions:
1. What country are you investing in?
2. What is your investment objective (capital appreciation, income, growth + income, speculation)?
3. Investment time horizon (days, months, 1–3 years, 3–10 years, 10+ years)?
4. Risk tolerance (very low, low, moderate, high, very high)?

Do NOT skip these questions. Wait for the user's answers before providing any analysis."""

# --- Page config ---
st.set_page_config(
    page_title="StockAdvisorAI",
    page_icon="📈",
    layout="centered",
)

st.title("📈 StockAdvisorAI")
st.caption("Your AI-powered stock research assistant")

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []       # what we show in the UI
    st.session_state.api_messages = []   # what we actually send to the API
    st.session_state.first_message = True

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if prompt := st.chat_input("Ask about stocks, markets, or investment ideas..."):
    # Show user message (always the clean version)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepend instructions on the first message (workaround for search model ignoring system prompt)
    if st.session_state.first_message:
        api_user_msg = {"role": "user", "content": f"{INSTRUCTIONS}\n\nUser question: {prompt}"}
        st.session_state.first_message = False
    else:
        api_user_msg = {"role": "user", "content": prompt}

    st.session_state.api_messages.append(api_user_msg)

    # Build full API messages
    api_messages = [{"role": "developer", "content": SYSTEM_PROMPT}] + st.session_state.api_messages

    # Get response from the model
    with st.chat_message("assistant"):
        with st.spinner("Searching & thinking..."):
            completion = client.chat.completions.create(
                model=MODEL,
                messages=api_messages,
                **MODEL_KWARGS,
            )
            response = completion.choices[0].message.content
        st.markdown(response)

    # Save assistant response to both UI and API message histories
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.api_messages.append({"role": "assistant", "content": response})
