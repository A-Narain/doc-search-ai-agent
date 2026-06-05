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

app = FastAPI()

UPLOAD_FOLDER = "uploads"

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt",
    ".xlsx", ".pptx", ".msg",
    ".html", ".md"
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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


class QuestionRequest(BaseModel):
    question: str
    filename_filter: str | None = None


@app.post("/chat")
async def chat(request: QuestionRequest):

    # ── Agentic retrieval loop ────────────────────────────
    # Handles query rewriting + iterative re-retrieval
    # until confidence threshold is met or max iterations reached
    retrieved_chunks, confidence, iteration_log = agentic_retrieve(
        request.question,
        filename_filter=request.filename_filter
    )

    if not retrieved_chunks:
        return {
            "question":         request.question,
            "answer":           "No relevant documents found. Please upload relevant files first.",
            "confidence_score": 0.0,
            "confidence_label": "None",
            "sources":          []
        }

    # ── Generate answer ───────────────────────────────────
    answer = generate_answer(request.question, retrieved_chunks)

    # ── Verify answer against retrieved context ───────────
    verification = verify_answer(request.question, answer, retrieved_chunks)

    # ── Format with citations, confidence label, agent trace
    return format_response(
        question=request.question,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        confidence=confidence,
        verification=verification,
        iteration_log=iteration_log
    )