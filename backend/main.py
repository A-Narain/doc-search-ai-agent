from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from document_processor import extract_text
from github_service import upload_to_github
from chunking import chunk_text
from vector_store import store_chunks, collection
from intent_classifier import classify_intent
from conversation_memory import (
    add_message,
    get_history_as_text,
    clear_session
)
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


# ── Health check ──────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "DocuMind AI — Agentic RAG Running"}


# ── Upload ────────────────────────────────────────────────

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(filepath, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

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


# ── Chat ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question:   str
    session_id: str = "default"


@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    user_msg   = request.question

    # 1. Get conversation history for this session
    conversation_history = get_history_as_text(session_id)

    # 2. Get available files from vector store
    result          = collection.get(include=["metadatas"])
    available_files = list({m["filename"] for m in result.get("metadatas", [])})

    # 3. Classify intent (passes history + available files for context)
    classified = classify_intent(user_msg, conversation_history, available_files)

    print(f"[Chat] Session: {session_id} | Intent: {classified['intent']} | Files: {classified['target_files']}")

    # 4. Store user message in memory
    add_message(session_id, "user", user_msg)

    # 5. Route through the full agentic pipeline
    result = route(classified, user_msg, conversation_history)

    # 6. Store assistant response in memory
    add_message(
        session_id,
        "assistant",
        result.get("answer", ""),
        metadata={"intent": result.get("intent"), "sources": result.get("sources", [])}
    )

    return result


# ── Session management ────────────────────────────────────

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared."}