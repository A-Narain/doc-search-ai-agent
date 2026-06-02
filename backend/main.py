from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os

from document_processor import extract_text
from github_service import upload_to_github
from chunking import chunk_text
from vector_store import store_chunks
from retriever import retrieve_chunks
from gemini_service import generate_answer
from query_rewriter import rewrite_query
from response_verifier import verify_answer

app = FastAPI()

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


@app.get("/")
def home():
    return {
        "message": "Document Search AI Agent Running"
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(filepath, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # Upload document to GitHub
    upload_to_github(
        filepath,
        file.filename
    )

    # Extract text
    text = extract_text(
        filepath
    )

    # Chunk text
    chunks = chunk_text(
        text
    )

    # Store embeddings
    store_chunks(
        chunks,
        file.filename
    )

    print(
        f"Stored {len(chunks)} chunks"
    )

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "chunks_stored": len(chunks)
    }


class QuestionRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(
    request: QuestionRequest
):

    # Query rewriting
    rewritten_query = rewrite_query(
        request.question
    )

    print(
        f"\nOriginal Query: {request.question}"
    )

    print(
        f"Rewritten Query: {rewritten_query}"
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks(
        rewritten_query
    )

    # Generate answer
    answer = generate_answer(
        request.question,
        retrieved_chunks
    )

    # Verify answer
    verification_result = verify_answer(
        request.question,
        answer,
        retrieved_chunks
    )

    # Build source list
    sources = []

    for chunk in retrieved_chunks:

        sources.append(
            {
                "filename": chunk["filename"],
                "chunk_id": chunk["chunk_id"]
            }
        )

    return {
        "question": request.question,
        "rewritten_query": rewritten_query,
        "answer": answer,
        "verification": verification_result,
        "sources": sources
    }