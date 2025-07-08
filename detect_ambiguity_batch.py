import os
from pathlib import Path
from core.embedding import split_into_sentence_chunks
from core.milvus_utilis import search_similar_chunks
from cli_app import extract_text_from_pdf, extract_text_from_txt
import openai
import sys
import nltk
import uuid
import datetime
from core.rag_chain import ask_llm_with_context_custom_prompt

# --- CONFIG ---
OUTPUT_DIR = 'suspicious_summaries'
CHUNK_SIZE = 700  # chars
OVERLAP = 1  # sentences
TOP_K_MILVUS = 3
OPENAI_MODEL = 'gpt-3.5-turbo'
MAX_CONTEXT_CHARS = 20000  # Conservative limit to avoid exceeding token context

# --- Ensure output dir exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load OpenAI API key ---
from config.config import OPENAI_API_KEY
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Ambiguity check prompt with context ---
AMBIGUITY_PROMPT = (
    "Is the following legal text ambiguous? If yes, explain why. If not, reply 'Not ambiguous.'\n\n" +
    "Text: \n{chunk}\n\n" +
    "Relevant Context:\n{context}"
)

def check_ambiguity(chunk, context):
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": AMBIGUITY_PROMPT.format(chunk=chunk, context=context)}],
        temperature=0.0,
        max_tokens=256
    )
    return response.choices[0].message.content.strip()

def process_file(file_path: Path):
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
    # Chunk at sentence boundaries
    chunks = split_into_sentence_chunks(text, target_chunk_size=CHUNK_SIZE, overlap_sentences=OVERLAP)
    print(f"{file_path.name}: {len(chunks)} chunks")
    suspicious_count = 0
    out_path = Path(OUTPUT_DIR) / (file_path.stem + '.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"# Ambiguity Summary for {file_path.name}\n\n")
        f.write(f"**Session ID:** {session_id}\n\n")
    for idx, chunk in enumerate(chunks):
        print(f"  Checking chunk {idx+1}/{len(chunks)}...")
        refs = search_similar_chunks(chunk, top_k=1000)
        context = "\n".join([ref['chunk'] for ref in refs])
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]
        prompt = f"""
You are a helpful assistant with access to document snippets.

Based on the following context, answer the user's question concisely and clearly:

CONTEXT:
{context}

QUESTION:
Is the following legal text ambiguous? If yes, explain why. If not, reply 'Not ambiguous.'\n\nText:\n{chunk}
"""
        answer = ask_llm_with_context_custom_prompt(prompt)
        if answer.lower().startswith('not ambiguous'):
            continue
        suspicious_count += 1
        with open(out_path, 'a', encoding='utf-8') as f:
            f.write(f"## Suspicious Chunk {suspicious_count}\n\n")
            f.write(f"**Chunk:**\n\n{chunk}\n\n")
            f.write(f"**LLM Explanation:**\n\n{answer}\n\n")
    print(f"\n✅ Done. {suspicious_count} ambiguous/suspicious chunks found. See {out_path}")

def main():
    file_path_str = input("Enter the path to the file you want to analyze (e.g., 'testing files/contract.txt'): ,WITHOUT QUOTES").strip()
    file_path = Path(file_path_str)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    process_file(file_path)

if __name__ == '__main__':
    main() 