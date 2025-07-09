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

CHUNK_SIZE = 700  # chars
OVERLAP = 1  # sentences
MILVUS_MAX_CHUNK_BYTES = 1000
OPENAI_MODEL = 'gpt-3.5-turbo'

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def split_long_chunk_bytes(chunk, max_bytes=MILVUS_MAX_CHUNK_BYTES):
    nltk.download('punkt', quiet=True)
    sentences = nltk.sent_tokenize(chunk)
    sub_chunks = []
    current = ''
    for sent in sentences:
        if len(sent.encode('utf-8')) > max_bytes:
            temp = ''
            for char in sent:
                if len((temp + char).encode('utf-8')) > max_bytes:
                    sub_chunks.append(temp)
                    temp = char
                else:
                    temp += char
            if temp:
                sub_chunks.append(temp)
            if current:
                sub_chunks.append(current)
                current = ''
        elif len((current + (' ' if current else '') + sent).encode('utf-8')) <= max_bytes:
            if current:
                current += ' '
            current += sent
        else:
            if current:
                sub_chunks.append(current)
            current = sent
    if current:
        sub_chunks.append(current)
    really_safe_chunks = []
    for c in sub_chunks:
        if len(c.encode('utf-8')) <= max_bytes:
            really_safe_chunks.append(c)
        else:
            temp = ''
            for char in c:
                if len((temp + char).encode('utf-8')) > max_bytes:
                    really_safe_chunks.append(temp)
                    temp = char
                else:
                    temp += char
            if temp:
                really_safe_chunks.append(temp)
    return really_safe_chunks

def enforce_max_chunk_byte_length(chunks, max_bytes=MILVUS_MAX_CHUNK_BYTES):
    safe_chunks = []
    for chunk in chunks:
        if len(chunk.encode('utf-8')) <= max_bytes:
            safe_chunks.append(chunk)
        else:
            safe_chunks.extend(split_long_chunk_bytes(chunk, max_bytes))
    for c in safe_chunks:
        assert len(c.encode('utf-8')) <= max_bytes, f"Chunk byte length {len(c.encode('utf-8'))} exceeds max {max_bytes}"
    return safe_chunks

# --- Multi-query function for Milvus law database ---
def multi_milvus_query(queries, top_k=5):
    """
    Given a list of law-related queries, return the top_k most relevant law context results for each query from the Milvus law database.
    The quality and specificity of your queries will directly impact the usefulness of the legal context you receive.
    Returns a dict: {query: [chunks]}
    """
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

    # Split contract into chunks
    chunks = split_into_sentence_chunks(text, target_chunk_size=CHUNK_SIZE, overlap_sentences=OVERLAP)
    chunks = enforce_max_chunk_byte_length(chunks, MILVUS_MAX_CHUNK_BYTES)
    print(f"{file_path.name}: {len(chunks)} chunks")

    out_path = Path(OUTPUT_DIR) / f"{file_path.stem}_{session_id}.md"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"# Ambiguity & Law Violation Summary for {file_path.name}\n\n")
        f.write(f"**Session ID:** {session_id}\n\n")

    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx+1}/{len(chunks)}...")
        system_prompt = (
            """
You are a legal expert tasked with reviewing contract clauses for ambiguity or potential violations of labor law.

Your primary goal is to determine, for each contract excerpt, whether it is ambiguous or violates any law. To do this, you should:

- Use MULTI_QUERY_LAW: [<query1>, <query2>, ...] to search the law database for relevant legal context. You may request multiple law queries in a single response. The quality and specificity of your questions to the law database (Milvus) are critical: well-formed, precise queries will yield much more useful legal context and lead to a better legal analysis.
- Only use QUERY_CONTRACT: <query> to search the previous contract content if you genuinely need to recall a definition or earlier clause that is not present in your current context (for example, if you have forgotten a term or need to check a cross-reference).

**Instructions:**
- Do NOT reply with acknowledgments, requests for time, or meta-comments. Begin your legal analysis and queries immediately.
- You may issue MULTI_QUERY_LAW or QUERY_CONTRACT as many times as needed. When you have enough information, reply with FINAL ANSWER: <your clear, concise, and referenced legal explanation and decision>.
- Your final answer should be professional, reference the law context you found, and be easy to understand for a non-lawyer.
            """
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTRACT EXCERPT:\n{chunk}\n\nBegin your legal analysis and queries as instructed above."}
        ]
        answer = None
        while True:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=512
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("MULTI_QUERY_LAW:"):
                # Parse queries from the format MULTI_QUERY_LAW: [q1, q2, ...]
                import ast
                queries_str = content[len("MULTI_QUERY_LAW:"):].strip()
                try:
                    queries = ast.literal_eval(queries_str)
                    if not isinstance(queries, list):
                        queries = [queries]
                except Exception:
                    queries = [queries_str]
                law_results = multi_milvus_query(queries, top_k=5)
                context_text = '\n\n'.join([f"Query: {q}\n" + '\n'.join(law_results[q]) for q in law_results])
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"LAW CONTEXT RESULT:\n{context_text}"})
            elif content.startswith("QUERY_CONTRACT:"):
                query = content[len("QUERY_CONTRACT:"):].strip()
                contract_context = search_contract_context(query, top_k=3)
                context_text = '\n'.join([ref['chunk'] for ref in contract_context])
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"CONTRACT CONTEXT RESULT:\n{context_text}"})
            elif content.startswith("FINAL ANSWER:"):
                answer = content[len("FINAL ANSWER:"):].strip()
                break
            else:
                answer = content
                break
        with open(out_path, 'a', encoding='utf-8') as f:
            f.write(f"## Chunk {idx+1}\n\n")
            f.write(f"**Chunk Text:**\n\n{chunk}\n\n")
            f.write(f"**LLM Explanation:**\n\n{answer}\n\n---\n\n")
        if len(chunk.encode('utf-8')) > MILVUS_MAX_CHUNK_BYTES:
            print(f"[FATAL] About to insert chunk of byte length {len(chunk.encode('utf-8'))}: {chunk[:60]}...")
            raise ValueError("Chunk too long in bytes!")
        save_to_contract_context([chunk], file_path.name)

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