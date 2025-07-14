import streamlit as st
from pathlib import Path
import tempfile
import time
import glob

from detect_ambiguity_batch import process_file_with_llm
from core.milvus_utilis import delete_all_contract_context
from core.rag_chain import ask_question_smart_with_toolcall

st.set_page_config(page_title="Contract Labor Law Analyzer", layout="wide")
st.title("📄 Contract Labor Law Analyzer")

st.markdown("""
Upload a contract (PDF or TXT) and get an AI-powered **labor law compliance analysis**. The app will:
- Check each clause for clear labor law violations
- Summarize all detected labor law violations
- Let you download a detailed Markdown report
""")

# --- Contract Upload & Analysis ---
st.header("1️⃣ Upload and Analyze Contract")
uploaded_file = st.file_uploader("Upload contract (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    if st.button("Analyze Contract", type="primary"):
        with st.spinner("Analyzing contract for labor law compliance..."):
            delete_all_contract_context()  # Clear previous contract context
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = Path(tmp_file.name)

            process_file_with_llm(temp_path)
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
    if analysis_content.strip():
        st.markdown(analysis_content)
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
            import re
            final_answer = ask_question_smart_with_toolcall(user_input)  # Direct output, no parsing
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
            st.markdown(
                f"<div style='background:#e0e0e0;color:#222;padding:8px;border-radius:8px;margin-bottom:8px'><b>AI:</b> {msg}</div>",
                unsafe_allow_html=True
            )
