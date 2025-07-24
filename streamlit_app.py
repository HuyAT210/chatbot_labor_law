import streamlit as st
from pathlib import Path
import tempfile
import time
import glob
import re
import markdown as md_lib

from detect_ambiguity_batch import process_file_with_llm
from core.rag_chain import ask_llm
from core.rag_chain import deep_search_pipeline, deep_search_pipeline_stream
from cli_app import extract_text_from_pdf, extract_text_from_txt

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
        with st.spinner("Analyzing contract... This may take a moment."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = Path(tmp_file.name)

            # --- Automatic analysis mode selection ---
            text = ""
            try:
                if temp_path.suffix.lower() == '.pdf':
                    text = extract_text_from_pdf(str(temp_path))
                elif temp_path.suffix.lower() == '.txt':
                    text = extract_text_from_txt(str(temp_path))
            except Exception as e:
                st.error(f"Error extracting text from file: {e}")
                st.stop()
            
            # The unified function will handle whether to chunk or not.
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

if "reasoning_histories" not in st.session_state:
    st.session_state.reasoning_histories = []

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a labor law question:", key="chat_input")
    submit_chat = st.form_submit_button("Send")

if submit_chat and user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("AI is thinking..."):
        try:
            def build_chat_history(history):
                lines = []
                for speaker, msg in history:
                    if speaker == "user":
                        lines.append(f"User: {msg}")
                    else:
                        lines.append(f"AI: {msg}")
                return "\n".join(lines)

            chat_history_str = build_chat_history(st.session_state.chat_history)
            # --- New persistent reasoning box for this question ---
            current_reasoning_steps = []
            # Do not append to reasoning_histories until finished
            final_answer = [None]
            def step_stream():
                for step in deep_search_pipeline_stream(user_input, chat_history=chat_history_str):
                    def strip_html_tags(text):
                        clean = re.sub(r'<[^>]+>', '', text)
                        return clean
                    if step["type"] == "subquestions":
                        content = '\n'.join([f'- {q}' for q in step["content"]])
                        yield f"**AI is expanding your question:**\n{content}"
                    elif step["type"] == "answer":
                        content = f"**AI is searching for:** {strip_html_tags(step['question'])}\n**Found:** {strip_html_tags(step['content'])}"
                        yield content
                    elif step["type"] == "quality_check":
                        if not step["accepted"]:
                            yield "**AI is refining its questions for deeper research...**"
                    elif step["type"] == "outline":
                        yield "**AI is organizing the answer outline...**"
                    elif step["type"] == "final_answer":
                        final_answer[0] = strip_html_tags(step["content"])
                        yield "**AI is writing the final answer...**"
                    elif step["type"] == "not_labor_law":
                        final_answer[0] = step["content"]
                        yield step["content"]
                    time.sleep(0.7)
            # Show only the current (live) reasoning box while thinking
            thinking_box = st.container()
            with thinking_box:
                reasoning_placeholder = st.empty()
                for streamed in step_stream():
                    current_reasoning_steps.append(streamed)
                    reasoning_markdown = '\n\n'.join(current_reasoning_steps)
                    reasoning_html = md_lib.markdown(reasoning_markdown, extensions=['extra', 'sane_lists'])
                    scrollable_box = (
                        f"<div style='max-height: 300px; overflow-y: auto; border: 1px solid #bbb; "
                        f"border-radius: 8px; padding: 12px; background: #e6e6e6; color: #222; margin-bottom: 12px;'>"
                        f"<b>Question:</b> {user_input}<br><br>{reasoning_html}</div>"
                    )
                    reasoning_placeholder.markdown(scrollable_box, unsafe_allow_html=True)
                time.sleep(0.5)
                reasoning_placeholder.empty()  # Remove the live box after finishing
            # Now append the finished reasoning to the histories
            st.session_state.reasoning_histories.append({
                "question": user_input,
                "steps": current_reasoning_steps
            })
            if final_answer[0] is not None:
                final_answer_str = re.sub(r'<think>.*?</think>', '', final_answer[0], flags=re.DOTALL)
                final_answer_str = re.sub(r'<think>.*?</think>', '', final_answer_str, flags=re.DOTALL)
            else:
                final_answer_str = "[ERROR] No answer generated."
        except Exception as e:
            final_answer_str = f"[ERROR] {e}"

    st.session_state.chat_history.append(("ai", final_answer_str))

# --- Render all completed reasoning boxes and chat bubbles grouped together ---
for i, history in enumerate(st.session_state.reasoning_histories):
    reasoning_markdown = '\n\n'.join(history["steps"])
    reasoning_html = md_lib.markdown(reasoning_markdown, extensions=['extra', 'sane_lists'])
    scrollable_box = (
        f"<div style='max-height: 300px; overflow-y: auto; border: 1px solid #bbb; "
        f"border-radius: 8px; padding: 12px; background: #e6e6e6; color: #222; margin-bottom: 4px;'>"
        f"<b>Question:</b> {history['question']}<br><br>{reasoning_html}</div>"
    )
    st.markdown(scrollable_box, unsafe_allow_html=True)

    # Render user and AI chat bubbles for this turn
    chat_idx = i * 2
    if chat_idx < len(st.session_state.chat_history):
        user_msg = st.session_state.chat_history[chat_idx][1]
        st.markdown(
            f"<div style='background:#e6f7ff;color:#222;padding:8px;border-radius:8px;margin-bottom:2px'><b>You:</b> {user_msg}</div>",
            unsafe_allow_html=True
        )
    if chat_idx + 1 < len(st.session_state.chat_history):
        ai_msg = st.session_state.chat_history[chat_idx+1][1]
        ai_msg_html = md_lib.markdown(ai_msg, extensions=['extra', 'sane_lists'])
        st.markdown(
            f"<div style='background:#f6f6f6;color:#222;padding:8px;border-radius:8px;margin-bottom:12px'><b>AI:</b> {ai_msg_html}</div>",
            unsafe_allow_html=True
        )
