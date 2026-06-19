from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os

from document_processor import extract_text
from github_service import upload_to_github
from chunking import chunk_text
from vector_store import store_chunks, collection
from intent_classifier import classify_intent
from conversation_memory import add_message, get_history_as_text, clear_session
from agent_router import route

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ── UI ────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    return FileResponse("../frontend/index.html")

@app.get("/health")
def health():
    return {"message": "DocuMind AI — Agentic RAG Running"}


# ── Upload ────────────────────────────────────────────────

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    upload_to_github(filepath, file.filename)
    text   = extract_text(filepath)
    chunks = chunk_text(text)
    store_chunks(chunks, file.filename)

    print(f"[Upload] Stored {len(chunks)} chunks for '{file.filename}'")
    return {
        "message":       "File uploaded successfully",
        "filename":      file.filename,
        "chunks_stored": len(chunks)
    }


# ── Documents ─────────────────────────────────────────────

@app.get("/documents")
def list_documents():
    result = collection.get(include=["metadatas"])
    files  = sorted({m["filename"] for m in result.get("metadatas", []) if m.get("filename")})
    return {"documents": files}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    result = collection.get(include=["metadatas"])
    ids_to_delete = [
        result["ids"][i]
        for i, m in enumerate(result.get("metadatas", []))
        if m.get("filename") == filename
    ]
    if not ids_to_delete:
        raise HTTPException(status_code=404, detail=f"'{filename}' not found in index.")
    collection.delete(ids=ids_to_delete)
    print(f"[Delete] Removed {len(ids_to_delete)} chunks for '{filename}'")
    return {"message": f"'{filename}' removed.", "chunks_deleted": len(ids_to_delete)}


# ── Chat ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question:   str
    session_id: str = "default"
    filename:   Optional[str] = None


@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    user_msg   = request.question

    conversation_history = get_history_as_text(session_id)

    result_meta     = collection.get(include=["metadatas"])
    available_files = list({m["filename"] for m in result_meta.get("metadatas", [])})

    classified = classify_intent(
        user_msg,
        conversation_history,
        available_files,
        scoped_file=request.filename
    )

    print(f"[Chat] Session: {session_id} | Intent: {classified['intent']} | Files: {classified['target_files']}")

    add_message(session_id, "user", user_msg)
    result = route(classified, user_msg, conversation_history)
    add_message(
        session_id, "assistant",
        result.get("answer", ""),
        metadata={"intent": result.get("intent"), "sources": result.get("sources", [])}
    )
    return result


# ── Session ───────────────────────────────────────────────

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared."}


# ── Static files (keep this last) ─────────────────────────

app.mount("/static", StaticFiles(directory="../frontend"), name="static")