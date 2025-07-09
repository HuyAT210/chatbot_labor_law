import os
from pathlib import Path
from cli_app import extract_text_from_pdf, extract_text_from_txt
from core.embedding import split_into_sentence_chunks
from core.milvus_utilis import search_similar_chunks, save_to_contract_context, search_contract_context
import datetime
import nltk
import openai
from config.config import OPENAI_API_KEY

OUTPUT_DIR = 'suspicious_summaries'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MILVUS_MAX_CHUNK_BYTES = 1000
OPENAI_MODEL = 'gpt-4o-mini'

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Function schemas for OpenAI tool calling ---
openai_functions = [
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
            "description": "Search the contract context database for relevant content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The contract-related query to search for."},
                    "top_k": {"type": "integer", "description": "Number of top results to return.", "default": 3}
                },
                "required": ["query"]
            }
        }
    }
]

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

    for idx, sentence in enumerate(sentences):
        print(f"Processing sentence {idx+1}/{len(sentences)}...")
        system_prompt = (
            """
You are a legal expert tasked with reviewing contract clauses for ambiguity or potential violations of labor law.

Your primary goal is to determine, for each contract excerpt, whether it is ambiguous or violates any law. To do this, you should:

- Use the available tools to search the law database (multi_milvus_query) for relevant legal context. You may request multiple law queries in a single tool call. The quality and specificity of your questions to the law database (Milvus) are critical: well-formed, precise queries will yield much more useful legal context and lead to a better legal analysis.
- Only use the contract context search tool (search_contract_context) if you genuinely need to recall a definition or earlier clause that is not present in your current context (for example, if you have forgotten a term or need to check a cross-reference).

**Instructions:**
- Do NOT reply with acknowledgments, requests for time, or meta-comments. Begin your legal analysis and tool calls immediately.
- You may issue tool calls as many times as needed. When you have enough information, reply with FINAL ANSWER: <your clear, concise, and referenced legal explanation and decision>.
- Do NOT output tool calls as your final answer. Only use these to request information, and always finish with FINAL ANSWER: ...
- Your final answer should be professional, reference the law context you found, and be easy to understand for a non-lawyer.
            """
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTRACT EXCERPT:\n{sentence}\n\nBegin your legal analysis and queries as instructed above."}
        ]
        answer = None
        while True:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=10000,
                tools=openai_functions,
                tool_choice="auto"
            )
            msg = response.choices[0].message
            content = msg.content.strip() if msg.content else ""
            # Handle tool calls
            if msg.tool_calls:
                tool_messages = []
                import json
                for tool_call in msg.tool_calls:
                    if tool_call.function.name == "multi_milvus_query":
                        params = json.loads(tool_call.function.arguments)
                        queries = params["queries"]
                        top_k = params.get("top_k", 5)
                        law_results = multi_milvus_query(queries, top_k=top_k)
                        context_text = '\n\n'.join([f"Query: {q}\n" + '\n'.join(law_results[q]) for q in law_results])
                        tool_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "multi_milvus_query",
                            "content": context_text
                        })
                    elif tool_call.function.name == "search_contract_context":
                        params = json.loads(tool_call.function.arguments)
                        query = params["query"]
                        top_k = params.get("top_k", 3)
                        contract_context = search_contract_context(query, top_k=top_k)
                        context_text = '\n'.join([ref['chunk'] for ref in contract_context])
                        tool_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "search_contract_context",
                            "content": context_text
                        })
                messages.append({"role": "assistant", "content": content, "tool_calls": msg.tool_calls})
                messages.extend(tool_messages)
                continue  # Continue the loop to let the LLM process the tool results
            # Handle final answer
            if content.upper().startswith("FINAL ANSWER:"):
                answer = content[len("FINAL ANSWER:"):].strip()
                break
            # Otherwise, treat as answer
            answer = content
            break
        # Only write after the loop, when answer is final
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

    print(f"\n✅ Done. See {out_path}")

def main():
    file_path_str = input("Enter the path to the file you want to analyze (e.g., 'testing files/contract.txt'): ,WITHOUT QUOTES : ").strip()
    file_path = Path(file_path_str)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    process_file_with_llm(file_path)

if __name__ == '__main__':
    main() 