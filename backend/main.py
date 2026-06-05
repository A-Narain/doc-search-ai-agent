from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os

from document_processor import extract_text
from github_service import upload_to_github
from chunking import chunk_text
from vector_store import store_chunks
from gemini_service import generate_answer
from response_verifier import verify_answer
from agent_loop import agentic_retrieve

app = FastAPI(
    title="Document Search AI Agent",
    version="1.0.0"
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/", summary="Health check")
def home():
    return {"message": "Document Search AI Agent Running"}


@app.post("/upload", summary="Upload a document")
async def upload_document(file: UploadFile = File(...)):

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(filepath, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    try:
        upload_to_github(filepath, file.filename)
    except Exception as e:
        print(f"[GitHub] Upload warning: {e}")

    try:
        text = extract_text(filepath)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    chunks = chunk_text(text)
    store_chunks(chunks, file.filename)

    print(f"[Upload] Stored {len(chunks)} chunks for {file.filename}")

    return {
        "message":       "File uploaded successfully",
        "filename":      file.filename,
        "chunks_stored": len(chunks)
    }


class QuestionRequest(BaseModel):
    question: str
    filename: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is the research proposal about?",
                    "filename": "Research Proposal.pdf"
                }
            ]
        }
    }


@app.post("/chat", summary="Ask a question")
async def chat(request: QuestionRequest):

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        retrieved_chunks, confidence, iteration_log = agentic_retrieve(
            request.question,
            filename_filter=request.filename
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Retrieval error: {str(e)}")

    try:
        answer = generate_answer(request.question, retrieved_chunks)
        verification_result = verify_answer(request.question, answer, retrieved_chunks)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Gemini error: {str(e)}")

    sources = [
        {
            "filename": chunk["filename"],
            "chunk_id": chunk["chunk_id"],
            "score":    chunk["score"]
        }
        for chunk in retrieved_chunks
    ]

    return {
        "question":         request.question,
        "answer":           answer,
        "verification":     verification_result,
        "confidence":       round(confidence, 3),
        "agent_iterations": iteration_log,
        "sources":          sources
    }


@app.get("/documents", summary="List all documents in the database")
def list_documents():
    from vector_store import collection
    results   = collection.get(include=["metadatas"])
    filenames = list({m["filename"] for m in results["metadatas"]})
    return {"documents": filenames, "count": len(filenames)}


@app.delete("/clear", summary="Clear all documents from the database")
def clear_database():
    from vector_store import client as chroma_client
    chroma_client.delete_collection("documents")
    chroma_client.get_or_create_collection("documents")
    return {"message": "All documents cleared from database"}