from sentence_transformers import SentenceTransformer
import nltk
from typing import List
import time

print("🕒 Starting model loading...")
start_time = time.time()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

load_time = time.time() - start_time
print(f"✅ Model loaded in {load_time:.2f} seconds")

# Chunk configuration
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
BATCH_SIZE = 32  # Process 32 chunks at a time

def split_into_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    # Split text into overlapping chunks
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += size - overlap
    return chunks

def split_into_sentence_chunks(text: str, target_chunk_size: int = 700, overlap_sentences: int = 1) -> List[str]:
    """
    Split text into chunks at sentence boundaries, aiming for target_chunk_size (in chars).
    Each chunk contains whole sentences and may overlap by a number of sentences.
    """
    nltk.download('punkt', quiet=True)
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) > target_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Overlap: keep last N sentences
            if overlap_sentences > 0:
                current_chunk = current_chunk[-overlap_sentences:]
                current_len = sum(len(s) for s in current_chunk)
            else:
                current_chunk = []
                current_len = 0
        current_chunk.append(sent)
        current_len += len(sent)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    start_time = time.time()
    all_embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        # Only show progress bar for large batches
        show_progress = len(batch) > 10
        batch_embeddings = model.encode(batch, show_progress_bar=show_progress)
        all_embeddings.extend(batch_embeddings.tolist())
    
    embed_time = time.time() - start_time
    print(f"⏱️ Embedding {len(chunks)} chunks took {embed_time:.2f} seconds")
    return all_embeddings
