from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# After all your routes, at the bottom of main.py:
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")


from fastapi.middleware.cors import CORSMiddleware


from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os

from document_processor import extract_text
from github_service import upload_to_github
from chunking import chunk_text
from vector_store import store_chunks, collection
from gemini_service import generate_answer
from response_verifier import verify_answer
from agent_loop import agentic_retrieve
from response_formatter import format_response
from edit_service import (
    identify_chunks_to_edit,
    apply_edit_to_chunk,
    rebuild_document_with_edits
)
from conversation_memory import add_message, get_history_as_text, clear_session
from intent_classifier import classify_intent
from agent_router import route

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_FOLDER = "uploads"

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt",
    ".xlsx", ".pptx", ".msg",
    ".html", ".md"
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ── Shared helper ─────────────────────────────────────────

def delete_document_chunks(filename: str) -> int:
    """Delete all ChromaDB chunks belonging to filename. Returns count deleted."""
    result = collection.get(
        where={"filename": filename},
        include=["metadatas"]
    )
    ids = result.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


@app.get("/")
def home():
    return {"message": "Document Search AI Agent Running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. "
                   f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(filepath, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # Upload document to GitHub
    try:
        upload_to_github(filepath, file.filename)
    except Exception as e:
        print(f"[GitHub] Upload failed for {file.filename}: {e}")

    # Extract text
    text = extract_text(filepath)

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract any text from '{file.filename}'."
        )

    # Chunk text
    chunks = chunk_text(text)

    # Store embeddings
    store_chunks(chunks, file.filename)

    print(f"Stored {len(chunks)} chunks for '{file.filename}'")

    return {
        "message":       "File uploaded successfully",
        "filename":      file.filename,
        "chunks_stored": len(chunks)
    }


@app.get("/documents")
def list_documents():
    """List all documents currently stored in the vector store."""
    result = collection.get(include=["metadatas"])
    metadatas = result.get("metadatas", [])

    seen = {}
    for meta in metadatas:
        fname = meta["filename"]
        seen[fname] = seen.get(fname, 0) + 1

    documents = [
        {"filename": fname, "chunks": count}
        for fname, count in sorted(seen.items())
    ]

    return {
        "total_documents": len(documents),
        "documents":       documents
    }


@app.get("/documents/{filename}/content")
def get_document_content(filename: str):
    """Return the extracted plain text of a stored document for editing."""
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404,
            detail=f"File '{filename}' not found on disk. It may have been deleted."
        )

    try:
        text = extract_text(filepath)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "filename": filename,
        "content":  text
    }


class ReindexRequest(BaseModel):
    content: str  # edited plain text submitted by the user


@app.put("/documents/{filename}/reindex")
def reindex_document(filename: str, request: ReindexRequest):
    """
    Replace a document's index with edited content.

    Flow:
      1. Validate the new content isn't empty
      2. Delete all existing ChromaDB chunks for this file
      3. Overwrite the file on disk with the new content
      4. Chunk + re-embed the new content
      5. Push updated file to GitHub
    """
    if not request.content.strip():
        raise HTTPException(
            status_code=400,
            detail="Content cannot be empty."
        )

    # 1. Check the document exists in the vector store
    result = collection.get(
        where={"filename": filename},
        include=["metadatas"]
    )
    if not result.get("ids"):
        raise HTTPException(
            status_code=404,
            detail=f"No indexed document found with filename '{filename}'."
        )

    # 2. Delete old chunks
    deleted = delete_document_chunks(filename)
    print(f"[Reindex] Deleted {deleted} old chunks for '{filename}'")

    # 3. Overwrite file on disk with edited plain text
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(request.content)

    # 4. Chunk + re-embed
    chunks = chunk_text(request.content)
    store_chunks(chunks, filename)
    print(f"[Reindex] Stored {len(chunks)} new chunks for '{filename}'")

    # 5. Push updated file to GitHub
    try:
        upload_to_github(filepath, filename)
    except Exception as e:
        print(f"[GitHub] Reindex push failed for '{filename}': {e}")

    return {
        "message":          f"'{filename}' reindexed successfully.",
        "filename":         filename,
        "chunks_deleted":   deleted,
        "chunks_stored":    len(chunks)
    }


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Remove a document from the index and from disk."""
    deleted = delete_document_chunks(filename)

    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No indexed document found with filename '{filename}'."
        )

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    return {
        "message":        f"'{filename}' deleted successfully.",
        "chunks_deleted": deleted
    }


class QuestionRequest(BaseModel):
    question:   str
    session_id: str = "default"   # client sends a UUID per conversation


@app.post("/chat")
async def chat(request: QuestionRequest):

    # 1. Load conversation history for this session
    conversation_history = get_history_as_text(request.session_id)

    # 2. Get list of available documents for intent resolution
    meta_result = collection.get(include=["metadatas"])
    available_files = list({
        m["filename"] for m in meta_result.get("metadatas", [])
    })

    # 3. Classify intent — what does the user actually want?
    classified = classify_intent(
        user_message=request.question,
        conversation_history=conversation_history,
        available_files=available_files
    )

    print(f"[Chat] Session: {request.session_id} | Intent: {classified['intent']}")

    # 4. Route to the correct tool based on intent
    response = route(
        classified_intent=classified,
        original_message=request.question,
        conversation_history=conversation_history
    )

    # 5. Update conversation memory
    add_message(request.session_id, "user", request.question)
    add_message(
        request.session_id,
        "assistant",
        response.get("answer", ""),
        metadata={"intent": classified["intent"], "sources": response.get("sources", [])}
    )

    # 6. Return response enriched with intent metadata
    return {
        **response,
        "intent":          classified["intent"],
        "session_id":      request.session_id,
        "refined_message": classified["refined_message"]
    }


@app.delete("/session/{session_id}")
def clear_conversation(session_id: str):
    """Clear conversation memory for a session."""
    clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared."}


class EditRequest(BaseModel):
    instruction: str        # e.g. "Change the refund period from 14 days to 30 days"
    filename: str           # which document to edit


@app.post("/edit")
async def edit_document(request: EditRequest):
    """
    AI-driven document edit.

    Flow:
      1. Retrieve relevant chunks using the edit instruction as the query
      2. Ask the LLM which chunks actually need changing
      3. Rewrite each affected chunk
      4. Splice rewrites back into the full document text
      5. Delete old chunks, store new ones, update file + GitHub
    """

    # 1. Retrieve relevant chunks from the target document
    retrieved_chunks, confidence, iteration_log = agentic_retrieve(
        request.instruction,
        filename_filter=request.filename
    )

    if not retrieved_chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No relevant content found in '{request.filename}' for that instruction."
        )

    # 2. Identify which chunks actually need editing
    chunks_to_edit = identify_chunks_to_edit(request.instruction, retrieved_chunks)

    if not chunks_to_edit:
        return {
            "message":    "No sections in the document matched the edit instruction.",
            "filename":   request.filename,
            "edits_made": 0
        }

    # 3. Rewrite each affected chunk
    edited = []
    for chunk in chunks_to_edit:
        rewritten = apply_edit_to_chunk(chunk["text"], request.instruction)
        edited.append({
            "filename":      chunk["filename"],
            "chunk_id":      chunk["chunk_id"],
            "original_text": chunk["text"],
            "rewritten_text": rewritten
        })

    # 4. Load the full document text and splice in rewrites
    filepath = os.path.join(UPLOAD_FOLDER, request.filename)
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404,
            detail=f"File '{request.filename}' not found on disk."
        )

    try:
        original_text = extract_text(filepath)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    updated_text = rebuild_document_with_edits(original_text, edited)

    # 5. Delete old chunks, store new ones
    deleted = delete_document_chunks(request.filename)

    # Overwrite file on disk with updated text
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(updated_text)

    new_chunks = chunk_text(updated_text)
    store_chunks(new_chunks, request.filename)

    # Push to GitHub
    try:
        upload_to_github(filepath, request.filename)
    except Exception as e:
        print(f"[GitHub] Edit push failed for '{request.filename}': {e}")

    return {
        "message":          f"'{request.filename}' edited and reindexed successfully.",
        "filename":         request.filename,
        "instruction":      request.instruction,
        "edits_made":       len(edited),
        "chunks_deleted":   deleted,
        "chunks_stored":    len(new_chunks),
        "changes": [
            {
                "chunk_id":  e["chunk_id"],
                "before":    e["original_text"][:200] + "..." if len(e["original_text"]) > 200 else e["original_text"],
                "after":     e["rewritten_text"][:200] + "..." if len(e["rewritten_text"]) > 200 else e["rewritten_text"]
            }
            for e in edited
        ]
    }