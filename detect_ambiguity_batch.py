import os
from pathlib import Path
from cli_app import extract_text_from_pdf, extract_text_from_txt
from core.milvus_utilis import search_similar_chunks, save_to_contract_context, search_contract_context
from core.milvus_utilis import delete_all_contract_context
import datetime
import nltk
import requests
import json
from config.config import QWEN_API_KEY, QWEN_API_URL, QWEN_MODEL
from violation_analyzer import is_clear_violation, generate_violation_summary
import time

OUTPUT_DIR = 'suspicious_summaries'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MILVUS_MAX_CHUNK_BYTES = 1000

# --- Function schemas for OpenAI tool calling ---
tool_functions = [
    {
        "type": "function",
        "function": {
            "name": "multi_milvus_query",
            "description": "Search the law database for relevant legal context using one or more queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of law-related queries to search for."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return per query.",
                        "default": 5
                    }
                },
                "required": ["queries"]
            }
        }
    }
]

def call_qwen_api(messages, tools=None, tool_choice="auto"):
    """
    Call Qwen API with the given messages and optional tools.
    """
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "top_p" : 1.0,
        "max_tokens": 32000
    }
    
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    
    response = requests.post(QWEN_API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Qwen API error: {response.status_code} - {response.text}")
    
    return response.json()



def split_document_into_sentences(text):
    nltk.download('punkt', quiet=True)
    sentences = nltk.sent_tokenize(text)
    # Ensure no sentence exceeds Milvus byte limit
    safe_sentences = []
    for sent in sentences:
        if len(sent.encode('utf-8')) <= MILVUS_MAX_CHUNK_BYTES:
            safe_sentences.append(sent)
        else:
            # Split long sentence by bytes
            temp = ''
            for char in sent:
                if len((temp + char).encode('utf-8')) > MILVUS_MAX_CHUNK_BYTES:
                    safe_sentences.append(temp)
                    temp = char
                else:
                    temp += char
            if temp:
                safe_sentences.append(temp)
    return safe_sentences

def multi_milvus_query(queries, top_k=5):
    results = {}
    for q in queries:
        refs = search_similar_chunks(q, top_k=top_k)
        results[q] = [ref['chunk'] for ref in refs]
    return results

# --- Main workflow ---
def process_file_with_llm(file_path: Path):
    total_start_time = time.time()
    session_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Session ID: {session_id}")
    # Extract text
    if file_path.suffix.lower() == '.pdf':
        text = extract_text_from_pdf(str(file_path))
    elif file_path.suffix.lower() == '.txt':
        text = extract_text_from_txt(str(file_path))
    else:
        print(f"Skipping unsupported file: {file_path.name}")
        return
    if not text.strip():
        print(f"No text extracted from {file_path.name}")
        return

    out_path = Path(OUTPUT_DIR) / f"{file_path.stem}_{session_id}.md"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"# Ambiguity & Law Violation Summary for {file_path.name}\n\n")
        f.write(f"**Session ID:** {session_id}\n\n")

    # --- WHOLE CONTRACT ANALYSIS ---
    system_prompt = (
        """
You are a legal compliance checker reviewing an entire contract to detect **all violations of labor law or employment standards, including the most obvious and trivial ones**.

You have access to the following tool:
- **multi_milvus_query**: Search for relevant legal rules and examples to confirm if something is illegal.

**Instructions:**
- For every clause or section, use the tool to check the relevant legal standard, even if you think you know the answer.
- List **every** violation, even if it is obvious, trivial, or a well-known rule (such as minimum wage, overtime pay, right to resign, paid leave, etc.).
- Err on the side of listing anything that could be a violation, even if it is common knowledge or seems minor.
- For each violation, provide the relevant contract excerpt and a short explanation.
- Do **NOT** skip any violation, even if it is obvious or trivial.
- If you are not sure, **assume it is a violation and list it**.
- If there are no clear violations, return an empty JSON array: []

**Output format:**
Output ONLY a valid JSON array of objects, where each object has two fields: 'excerpt' and 'explanation'.
Example:
[
  {"excerpt": "The employee will be paid less than minimum wage.", "explanation": "This violates minimum wage laws."},
  {"excerpt": "No overtime will be paid.", "explanation": "This violates overtime pay requirements."}
]
Do not include any text, markdown, or explanation outside the JSON array. Do not use markdown formatting. Do not add any commentary or headers.
"""
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"CONTRACT TEXT:\n{text}\n\nBegin your legal analysis now. Use tools if needed, then provide your final answer."}
    ]
    try:
        # Debug: print prompt and contract size
        print(f"[DEBUG] Prompt length: {len(system_prompt)} | Contract length: {len(text)}")
        if len(text) > 8000:
            print("[WARNING] Contract text is very large. The LLM may not process the entire document. Consider splitting the contract.")
        response = call_qwen_api(messages, tools=tool_functions, tool_choice="auto")
        print("[DEBUG] RAW API RESPONSE:", response)
        if 'choices' not in response or not response['choices']:
            print("[ERROR] No choices in API response.")
            return
        msg = response['choices'][0].get('message', {})
        print("[DEBUG] RAW MESSAGE:", msg)
        content = msg.get('content', '').strip() if msg.get('content') else ""
        print("[DEBUG] RAW CONTENT:", content)
        tool_calls = msg.get('tool_calls', [])
        # Handle tool calls if present
        if tool_calls:
            tool_messages = []
            for tool_call in tool_calls:
                function_name = tool_call['function']['name']
                function_args = tool_call['function']['arguments']
                tool_call_id = tool_call['id']
                if function_name == "multi_milvus_query":
                    params = json.loads(function_args)
                    queries = params["queries"]
                    top_k = params.get("top_k", 5)
                    law_results = multi_milvus_query(queries, top_k=top_k)
                    context_text = '\n\n'.join([f"Query: {q}\n" + '\n'.join(law_results[q]) for q in law_results])
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": "multi_milvus_query",
                        "content": context_text
                    })
            # Add the assistant's message and tool results to the conversation
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            messages.extend(tool_messages)
            # Call the LLM again with the tool results
            final_response = call_qwen_api(messages)
            print("[DEBUG] RAW FINAL RESPONSE:", final_response)
            final_msg = final_response['choices'][0]['message']
            final_content = final_msg.get('content', '').strip() if final_msg.get('content') else ""
            print("[DEBUG] RAW FINAL CONTENT:", final_content)
            content = final_content
        # Remove <think> tags and clean up
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<think>.*?', '', content, flags=re.DOTALL)
        content = content.strip('`').strip()
        # Write raw LLM output
        with open(out_path, 'a', encoding='utf-8') as f:
            f.write(f"**LLM Output:**\n\n{content}\n\n---\n\n")
            print(f"**LLM Output:**\n\n{content}\n\n---\n\n")
        # Parse violations as JSON only
        violations = []
        try:
            if not content or content.strip() == '[]':
                print("ℹ️ No violations found in this contract.")
            else:
                violations = json.loads(content)
                if not isinstance(violations, list):
                    print("[ERROR] LLM output is not a list.")
                    return
                # Validate each violation object
                violations = [v for v in violations if isinstance(v, dict) and 'excerpt' in v and 'explanation' in v and v['excerpt'].strip() and v['explanation'].strip()]
                if not violations:
                    print("ℹ️ No violations found in this contract.")
        except Exception as e:
            print(f"[ERROR] Failed to parse LLM output as JSON: {e}")
            return
        if violations:
            contract_name = file_path.stem
            summary = generate_violation_summary([(v['excerpt'], v['explanation']) for v in violations], contract_name)
            # Create violation report
            output_dir = 'violation_analysis'
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = Path(output_dir) / f"{contract_name}_violation_analysis_{timestamp}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# Violation Analysis for {contract_name}\n\n")
                f.write(f"**Analysis Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Total Violations Found:** {len(violations)}\n\n")
                f.write("## Violations Summary\n\n")
                f.write(summary)
                f.write("\n\n## Detailed Violations\n\n")
                for i, v in enumerate(violations, 1):
                    f.write(f"### Violation {i}\n\n")
                    f.write(f"**Excerpt:** {v['excerpt']}\n\n")
                    f.write(f"**Explanation:** {v['explanation']}\n\n")
                    f.write("---\n\n")
            print(f"✅ Violation report created: {output_file}")
    except Exception as e:
        print(f"[ERROR] Whole contract analysis: {e}")
    total_end_time = time.time()
    print(f"[TIMER] Total process_file_with_llm time: {total_end_time - total_start_time:.2f} seconds.")


def main():
    file_path_str = input("Enter the path to the file you want to analyze (e.g., 'testing files/contract.txt'): ,WITHOUT QUOTES : ").strip()
    file_path = Path(file_path_str)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    delete_all_contract_context()
    process_file_with_llm(file_path)
def test():
    final_response = call_qwen_api([{"role": "user", "content": "Tell me a joke"}])
    final_content = final_response['choices'][0]['message']['content'].strip()
    answer = final_content
    print(answer)
if __name__ == '__main__':
    main()