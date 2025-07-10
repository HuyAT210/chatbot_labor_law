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
    },
    {
        "type": "function",
        "function": {
            "name": "search_contract_context",
            "description": "Search the contract context database for relevant content using multiple queries to find contract clauses, terms, or definitions that may be referenced in the current analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of contract-related queries to search for, such as term definitions, clause references, or related contract sections."
                    },
                    "top_k": {"type": "integer", "description": "Number of top results to return per query.", "default": 3}
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
        "max_tokens": 10000
    }
    
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    
    response = requests.post(f"{QWEN_API_URL}/chat/completions",headers=headers, json=payload)
    
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

    # Split contract into sentences
    sentences = split_document_into_sentences(text)
    print(f"{file_path.name}: {len(sentences)} sentences")

    out_path = Path(OUTPUT_DIR) / f"{file_path.stem}_{session_id}.md"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"# Ambiguity & Law Violation Summary for {file_path.name}\n\n")
        f.write(f"**Session ID:** {session_id}\n\n")

    all_sentence_results = []

    for idx, sentence in enumerate(sentences):
        print(f"Processing sentence {idx+1}/{len(sentences)}...")
        system_prompt = (
            """
You are a legal compliance checker reviewing contract clauses to detect **clear, unambiguous violations of labor law or employment standards**.  
You must focus ONLY on contract sentences that clearly break the law.

⚠️ **CRITICAL**: You must provide a definitive answer. If the violation is unclear, uncertain, or requires additional context, you MUST respond with `NO CLEAR VIOLATION.`

For each contract excerpt:
- If the sentence clearly violates a labor law or mandatory employment regulation, give a **short, clear explanation of the violation in plain English**.
- If there is any ambiguity, uncertainty, or lack of clear violation, respond: `NO CLEAR VIOLATION.`

**Available tools (use as needed):**
- **multi_milvus_query**: Search for relevant legal rules and examples to confirm if something is illegal
- **search_contract_context**: Search for contract clauses, term definitions, or related sections that may be referenced

**Important:** You can use both tools, just one, or none at all in a single response. After using any tools, provide your final analysis.

**Output format:**  
You MUST start your final answer with: `FINAL ANSWER:` followed by your short explanation.  
Example:  
`FINAL ANSWER: This clause violates the Fair Labor Standards Act by failing to pay overtime.`  
or  
`FINAL ANSWER: NO CLEAR VIOLATION.`

🚫 Do not provide background explanations, summaries of the contract, or meta-comments.  
🚫 Do not speculate about potential issues or "may" scenarios.  
✅ Only give your conclusion about whether this sentence violates the law.

Begin your legal analysis now. Use tools if needed, then provide your final answer.
   """
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTRACT EXCERPT:\n{sentence}\n\nBegin your legal analysis now. Use tools if needed, then provide your final answer."}
        ]
        
        # Call Qwen API
        response = call_qwen_api(messages, tools=tool_functions, tool_choice="auto")
        msg = response['choices'][0]['message']
        content = msg.get('content', '').strip() if msg.get('content') else ""
        tool_calls = msg.get('tool_calls', [])
        
        # Handle tool calls if present
        if tool_calls:
            tool_messages = []
            for tool_call in tool_calls:
                # Qwen returns dictionary format
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
                elif function_name == "search_contract_context":
                    params = json.loads(function_args)
                    queries = params["queries"]
                    top_k = params.get("top_k", 3)
                    contract_context_results = search_contract_context(queries, top_k=top_k)
                    # Combine all results from all queries
                    all_context_chunks = []
                    for query, results in contract_context_results.items():
                        all_context_chunks.append(f"Query: {query}")
                        all_context_chunks.extend([ref['chunk'] for ref in results])
                    context_text = '\n\n'.join(all_context_chunks)
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": "search_contract_context",
                        "content": context_text
                    })
            
            # Add tool results and get final response
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            messages.extend(tool_messages)
            
            # Get final response after tool calls
            final_response = call_qwen_api(messages)
            final_content = final_response['choices'][0]['message']['content'].strip()
            answer = final_content
        else:
            # No tool calls, use content directly
            answer = content
        
        # Strip HTML-like elements and extract final answer
        if answer:
            # Remove <think> elements and their content
            import re
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
            answer = re.sub(r'<think>.*?', '', answer, flags=re.DOTALL)  # Handle unclosed tags
            
            # Extract final answer if it starts with "FINAL ANSWER:"
            if answer.upper().startswith("FINAL ANSWER:"):
                answer = answer[len("FINAL ANSWER:"):].strip()
        
        # Write results
        if answer:
            with open(out_path, 'a', encoding='utf-8') as f:
                f.write(f"## Sentence {idx+1}\n\n")
                f.write(f"**Sentence Text:**\n\n{sentence}\n\n")
                f.write(f"**LLM Explanation:**\n\n{answer}\n\n---\n\n")
        else:
            print(f"[WARNING] Skipping output for sentence {idx+1} because LLM did not provide a final answer.")
        if len(sentence.encode('utf-8')) > MILVUS_MAX_CHUNK_BYTES:
            print(f"[FATAL] About to insert sentence of byte length {len(sentence.encode('utf-8'))}: {sentence[:60]}...")
            raise ValueError("Sentence too long in bytes!")
        save_to_contract_context([sentence], file_path.name)

        all_sentence_results.append({
            "sentence": sentence,
            "explanation": answer
        })

    print(f"\n✅ Done. See {out_path}")

    # Analyze violations and create report using in-memory results
    violations = [r for r in all_sentence_results if is_clear_violation(r["explanation"])]
    if violations:
        contract_name = file_path.stem
        summary = generate_violation_summary(violations, contract_name)

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
            for i, violation in enumerate(violations, 1):
                f.write(f"### Violation {i}\n\n")
                f.write(f"**Sentence:** {violation['sentence']}\n\n")
                f.write(f"**Explanation:** {violation['explanation']}\n\n")
                f.write("---\n\n")
        print(f"✅ Violation report created: {output_file}")
    else:
        print("ℹ️ No violations found in this contract.")


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