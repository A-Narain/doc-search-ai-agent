from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os

from github_service import upload_to_github
from document_processor import extract_text
from chunking import chunk_text
from vector_store import store_chunks
from retriever import retrieve_chunks
from gemini_service import generate_answer

app = FastAPI()

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Document Search AI Agent Running"
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(filepath, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # Upload to GitHub
    upload_to_github(
        filepath,
        file.filename
    )

    # Extract text
    text = extract_text(filepath)

    # Chunk text
    chunks = chunk_text(text)

    # Store embeddings in ChromaDB
    store_chunks(
        chunks,
        file.filename
    )

    print(f"Stored {len(chunks)} chunks")

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "chunks_stored": len(chunks)
    }


@app.post("/chat")
async def chat(request: QuestionRequest):

    retrieved_chunks = retrieve_chunks(
        request.question
    )

    answer = generate_answer(
    request.question,
    retrieved_chunks
)

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
    "answer": answer,
    "sources": sources
    }