import time
import random
import threading
import chromadb
from sentence_transformers import SentenceTransformer

# ── Embedding model (loaded once, shared across threads) ──
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# ── ChromaDB client ───────────────────────────────────────
client = chromadb.PersistentClient(path="./chroma_db")

# ── Thread lock for SQLite write safety ───────────────────
_db_lock = threading.Lock()

# ── Default shared collection (for single-user / no session) ──
collection = client.get_or_create_collection(name="documents")


# ── FIX 1: Dynamic collection per session ────────────────
def get_collection(session_id: str = None):
    """
    Returns a collection scoped to a session_id.
    If no session_id, returns the default shared collection.

    This prevents data mixing between different users/sessions.
    """
    if not session_id:
        return collection  # default single-user collection

    collection_name = f"documents_{session_id}"
    return client.get_or_create_collection(name=collection_name)


# ── FIX 2: Thread-safe store with retry loop ─────────────
def store_chunks(chunks, filename, session_id: str = None, max_retries: int = 5):
    """
    Embeds and stores chunks into ChromaDB.

    - Uses session_id to isolate data per user/session.
    - Wraps collection.add() in a retry loop with randomized
      backoff to handle SQLite 'database is locked' errors.
    """
    target_collection = get_collection(session_id)

    embeddings = embedding_model.encode(chunks)

    ids = [f"{filename}_{i}" for i in range(len(chunks))]

    metadatas = [
        {"filename": filename, "chunk_id": i}
        for i in range(len(chunks))
    ]

    # ── Retry loop for SQLite lock safety ────────────────
    for attempt in range(1, max_retries + 1):
        try:
            with _db_lock:  # only one thread writes at a time
                target_collection.add(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas
                )
            print(f"[VectorStore] Stored {len(chunks)} chunks for '{filename}'"
                  + (f" (session: {session_id})" if session_id else ""))
            return  # success — exit

        except Exception as e:
            error_msg = str(e).lower()
            if "locked" in error_msg or "busy" in error_msg:
                wait = round(random.uniform(0.1, 0.5) * attempt, 3)
                print(f"[VectorStore] SQLite locked — retrying in {wait}s "
                      f"(attempt {attempt}/{max_retries})")
                time.sleep(wait)
            else:
                # Not a lock error — raise immediately
                raise

    raise RuntimeError(
        f"[VectorStore] Failed to store chunks for '{filename}' "
        f"after {max_retries} attempts due to SQLite lock."
    )