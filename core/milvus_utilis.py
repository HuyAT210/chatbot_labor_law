import chromadb
from core.embedding import embed_chunks, split_into_chunks
import uuid
import time
import os

print("--- ChromaDB Debugging ---")

persist_directory = os.path.abspath("./chroma_data")
print(f"🔍 ChromaDB Persist Directory (Absolute Path): {persist_directory}")

if os.path.exists(persist_directory):
    if not os.listdir(persist_directory):
        print("⚠️  Warning: Persist directory exists but is EMPTY.")
    else:
        print("✅ Persist directory exists and is NOT empty.")
else:
    print("ℹ️ Info: Persist directory does not exist. It will be created.")

print("🕒 Connecting to ChromaDB...")
start_time = time.time()

# Use PersistentClient for ChromaDB 1.x
chroma_client = chromadb.PersistentClient(path=persist_directory)

print(f"Chroma client: {chroma_client}")

_collection = chroma_client.get_or_create_collection(
    name="documents"
)

def get_collection():
    """Returns the current collection object."""
    return _collection

connect_time = time.time() - start_time
print(f"✅ Connected to ChromaDB in {connect_time:.2f} seconds")

try:
    count = _collection.count()
    print(f"📊 Collection 'documents' currently has {count} items.")
    if count == 0:
        print("⚠️ Warning: Collection is empty after loading.")
except Exception as e:
    print(f"❌ Error getting collection count: {e}")

print("--------------------------\n")

def save_to_chroma(chunks: list[str], filename: str, vectors: list[list[float]] | None = None):
    start_time = time.time()
    if vectors is None:
        vectors = embed_chunks(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"filename": filename} for _ in chunks]
    _collection.add(
        ids=ids,
        documents=chunks,
        embeddings=vectors,
        metadatas=metadatas
    )
    save_time = time.time() - start_time
    print(f"✅ Saved {len(chunks)} chunks to ChromaDB in {save_time:.2f} seconds")
    if os.path.exists(persist_directory):
        files = os.listdir(persist_directory)
        print(f"🗂️ Files in persist directory after save: {files}")
        if not files:
            print("❌ No files written to persist directory after save! Possible Chroma persistence issue.")
    else:
        print("❌ Persist directory does not exist after save!")

def search_similar_chunks(query: str, top_k: int = 1000):
    start_time = time.time()
    query_vectors = embed_chunks([query])
    if not query_vectors:
        raise ValueError("Failed to generate embedding for query")
    query_vector = query_vectors[0]
    results = _collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )
    matches = []
    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for doc, dist in zip(docs, dists):
        matches.append({
            "score": 1 - dist,
            "chunk": doc
        })
    search_time = time.time() - start_time
    print(f"⏱️ Search completed in {search_time:.2f} seconds")
    return matches

def delete_file(filename: str):
    results = _collection.get(
        where={"filename": filename},
        include=["ids"]
    )
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        _collection.delete(ids=ids_to_delete)
        print(f"✅ Deleted all chunks of {filename} from ChromaDB.")
        return {
            "filename": filename,
            "message": f"✅ Deleted all chunks of {filename} from ChromaDB."
        }
    else:
        print(f"📭 No data to delete for {filename}.")
        return {
            "filename": filename,
            "message": f"📭 No data to delete for {filename}."
        }

def delete_all():
    try:
        global _collection
        chroma_client.delete_collection("documents")
        _collection = chroma_client.get_or_create_collection(
            name="documents"
        )
        print(f"🗑️ Deleted all records from ChromaDB collection.")
        return {
            "message": f"✅ Successfully deleted all records from the database."
        }
    except Exception as e:
        print(f"❌ Error deleting all data: {e}")
        return {
            "message": f"❌ Error deleting all data: {e}"
        }
