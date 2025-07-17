import streamlit as st
from pathlib import Path
import tempfile
import time
import glob
import re
import markdown as md_lib

from detect_ambiguity_batch import process_file_with_llm
from core.milvus_utilis import delete_all_contract_context
from core.rag_chain import ask_llm
from core.rag_chain import deep_search_pipeline

st.set_page_config(page_title="Contract Labor Law Analyzer", layout="wide")
st.title("📄 Contract Labor Law Analyzer")

st.markdown("""
Upload a contract (PDF or TXT) and get an AI-powered **labor law compliance analysis**. The app will:
- Check each clause for clear labor law violations
- Summarize all detected labor law violations
- Let you download a detailed Markdown report
""")

def fix_markdown(md):
    # Ensure two newlines before headings and lists
    md = re.sub(r'([^\n])(\n#)', r'\1\n\n#', md)
    md = re.sub(r'([^\n])(\n- )', r'\1\n\n- ', md)
    # Remove excessive blank lines (more than 2)
    md = re.sub(r'\n{3,}', '\n\n', md)
    # Remove line breaks in the middle of words (e.g., '7.25\n/\nh\no\nu\nr')
    md = re.sub(r'(\w)\n(\w)', r'\1\2', md)
    # Replace single newlines (not between paragraphs) with a space
    md = re.sub(r'(?<!\n)\n(?!\n)', ' ', md)
    # Optionally, collapse multiple spaces
    md = re.sub(r' +', ' ', md)
    return md.strip()

def render_markdown(md_text):
    # Normalize line endings
    md_text = md_text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove trailing spaces on each line
    md_text = '\n'.join(line.rstrip() for line in md_text.split('\n'))
    # Remove excessive blank lines (more than 2)
    while '\n\n\n' in md_text:
        md_text = md_text.replace('\n\n\n', '\n\n')
    # Convert markdown to HTML
    html = md_lib.markdown(md_text, extensions=['extra', 'sane_lists'])
    return html

# --- Contract Upload & Analysis ---
st.header("1️⃣ Upload and Analyze Contract")
uploaded_file = st.file_uploader("Upload contract (PDF or TXT)", type=["pdf", "txt"])

# New: Add a box for user to type a question about the contract
user_contract_question = st.text_input(
    "What do you want to know about this contract? (e.g., Are there any risks? Is the non-compete clause enforceable? Summarize the obligations, etc.)",
    key="contract_question_input"
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    if st.button("Analyze Contract", type="primary"):
        with st.spinner("Analyzing contract for your question..."):
            delete_all_contract_context()  # Clear previous contract context
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = Path(tmp_file.name)

            # Pass the user's question to the backend analysis
            from detect_ambiguity_batch import process_file_with_llm
            process_file_with_llm(temp_path, user_question=user_contract_question)
            time.sleep(1)  # Ensure file is written

            base_stem = temp_path.stem
            analysis_files = sorted(
                glob.glob(f"violation_analysis/{base_stem}_violation_analysis_*.md"),
                reverse=True
            )

            if analysis_files:
                st.session_state["last_analysis_file"] = analysis_files[0]
            else:
                st.session_state["last_analysis_file"] = None

if "last_analysis_file" in st.session_state and st.session_state["last_analysis_file"]:
    analysis_file = st.session_state["last_analysis_file"]
    with open(analysis_file, "r", encoding="utf-8") as f:
        analysis_content = f.read()

    st.subheader("Labor Law Violation Analysis Report")
    # If there is no analysis file, it means no violations were found
    if not ("last_analysis_file" in st.session_state and st.session_state["last_analysis_file"]):
        st.markdown("✅ No labor law violations were found in the uploaded contract.")
    else:
        analysis_file = st.session_state["last_analysis_file"]
        with open(analysis_file, "r", encoding="utf-8") as f:
            analysis_content = f.read()
        if analysis_content.strip():
            st.markdown(render_markdown(analysis_content), unsafe_allow_html=True)
            st.download_button(
                "Download Labor Law Violation Report",
                analysis_content,
                file_name=Path(analysis_file).name,
                mime="text/markdown"
            )
        else:
            st.markdown("✅ No labor law violations were found in the uploaded contract.")

# --- Chat Interface ---
st.header("2️⃣ Chat about Labor Law")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a labor law question:", key="chat_input")
    submit_chat = st.form_submit_button("Send")

if submit_chat and user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("AI is thinking..."):
        try:
            # Build chat history string for context
            def build_chat_history(history):
                lines = []
                for speaker, msg in history:
                    if speaker == "user":
                        lines.append(f"User: {msg}")
                    else:
                        lines.append(f"AI: {msg}")
                return "\n".join(lines)

            chat_history_str = build_chat_history(st.session_state.chat_history)
            final_answer = deep_search_pipeline(user_input, chat_history=chat_history_str)
            final_answer = re.sub(r'<think>.*?</think>', '', final_answer, flags=re.DOTALL)
            final_answer = re.sub(r'<think>.*?</think>', '', final_answer, flags=re.DOTALL)
        except Exception as e:
            final_answer = f"[ERROR] {e}"

    st.session_state.chat_history.append(("ai", final_answer))

if st.session_state.chat_history:
    for speaker, msg in st.session_state.chat_history:
        if speaker == "user":
            st.markdown(
                f"<div style='background:#e6f7ff;color:#222;padding:8px;border-radius:8px;margin-bottom:4px'><b>You:</b> {msg}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(render_markdown(msg), unsafe_allow_html=True)
