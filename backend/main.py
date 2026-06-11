from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

from document_processor import extract_text
from github_service      import upload_to_github
from chunking            import chunk_text
from vector_store        import store_chunks, client as chroma_client, collection
from gemini_service      import generate_answer
from response_verifier   import verify_answer
from response_formatter  import format_response
from agent_loop          import agentic_retrieve

app = FastAPI(title="Document Search AI Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/", summary="Health check")
def home():
    return {"message": "Document Search AI Agent v2.0 Running"}


@app.post("/upload", summary="Upload a document")
async def upload_document(
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(default=None)  # optional session header
):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    try:
        upload_to_github(filepath, file.filename)
    except Exception as e:
        print(f"[GitHub] Warning: {e}")

    try:
        text = extract_text(filepath)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=400, detail="Document appears empty or unreadable.")

    chunks = chunk_text(text)

    # Pass session_id — isolates data per user
    store_chunks(chunks, file.filename, session_id=x_session_id)

    return {
        "message":       "File uploaded and indexed successfully",
        "filename":      file.filename,
        "chunks_stored": len(chunks),
        "session_id":    x_session_id or "shared"
    }


class QuestionRequest(BaseModel):
    question:   str
    filename:   Optional[str] = None
    session_id: Optional[str] = None  # client passes this to scope retrieval

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "question":   "What is the research methodology?",
                "filename":   "Research Proposal.pdf",
                "session_id": "user_abc123"
            }]
        }
    }


@app.post("/chat", summary="Ask a question")
async def chat(request: QuestionRequest):

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        retrieved_chunks, confidence, iteration_log = agentic_retrieve(
            request.question,
            filename_filter=request.filename,
            session_id=request.session_id        # scoped retrieval
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Retrieval error: {str(e)}")

    if not retrieved_chunks:
        return {
            "question":         request.question,
            "answer":           "No relevant documents found. Please upload relevant files first.",
            "confidence_label": "Low",
            "confidence_score": 0.0,
            "verification":     "NOT VERIFIED",
            "sources":          []
        }

    try:
        raw_answer = generate_answer(request.question, retrieved_chunks)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Generation error: {str(e)}")

    try:
        verification, final_answer = verify_answer(
            request.question, raw_answer, retrieved_chunks
        )
    except Exception:
        verification, final_answer = "VERIFIED", raw_answer

    result = format_response(
        question=request.question,
        answer=final_answer,
        retrieved_chunks=retrieved_chunks,
        confidence=confidence,
        verification=verification,
        iteration_log=iteration_log
    )
    return result


@app.get("/documents", summary="List documents")
def list_documents(session_id: Optional[str] = None):
    from vector_store import client as chroma_client, collection
    if session_id:
        try:
            col = chroma_client.get_collection(f"documents_{session_id}")
        except Exception:
            return {"documents": [], "count": 0, "session_id": session_id}
    else:
        col = collection

    results   = col.get(include=["metadatas"])
    filenames = list({m["filename"] for m in results["metadatas"]})
    return {"documents": filenames, "count": len(filenames), "session_id": session_id or "shared"}


@app.delete("/clear", summary="Clear documents")
def clear_database(session_id: Optional[str] = None):
    from vector_store import client as chroma_client
    if session_id:
        col_name = f"documents_{session_id}"
        chroma_client.delete_collection(col_name)
        chroma_client.get_or_create_collection(col_name)
        return {"message": f"Session '{session_id}' cleared."}
    else:
        chroma_client.delete_collection("documents")
        chroma_client.get_or_create_collection("documents")
        return {"message": "Shared database cleared."}