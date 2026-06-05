import os
import time
from google import genai
from dotenv import load_dotenv

from retriever import retrieve_chunks
from query_rewriter import rewrite_query

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL                = "gemini-2.5-flash"
MAX_ITERATIONS       = 3
CONFIDENCE_THRESHOLD = 0.40


def compute_confidence(retrieved_chunks):
    if not retrieved_chunks:
        return 0.0
    scores = [chunk["score"] for chunk in retrieved_chunks]
    return sum(scores) / len(scores)


def generate_alternative_query(original_question, current_query, confidence):

    prompt = f"""
You are helping improve a document search query that returned poor results.

Original user question:
{original_question}

Current search query that gave low results:
{current_query}

Retrieval confidence score: {confidence:.2f} (scale 0.0 to 1.0)

Generate ONE alternative search query using different keywords or a different angle.
Return ONLY the new query. No explanation. No quotes.
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            if "503" in str(e) and attempt < 2:
                print(f"[Gemini] 503 error, retrying in 5s... (attempt {attempt+1})")
                time.sleep(5)
            else:
                raise


def agentic_retrieve(question, filename_filter=None):

    # Skip LLM query rewrite to save API calls — use question directly
    # Only rewrite if question is very short (under 10 words)
    if len(question.split()) < 10:
        try:
            query = rewrite_query(question)
        except Exception:
            query = question  # fallback to original if quota hit
    else:
        query = question

    best_chunks     = []
    best_confidence = 0.0
    iteration_log   = []

    # Reduce to 1 iteration to save quota — only retry if truly empty
    max_iter = 1

    for iteration in range(1, max_iter + 1):

        print(f"\n[Agent] Iteration {iteration}/{max_iter}")
        print(f"[Agent] Query: {query}")

        chunks     = retrieve_chunks(query, filename_filter=filename_filter)
        confidence = compute_confidence(chunks)

        print(f"[Agent] Confidence: {confidence:.3f}")

        iteration_log.append({
            "iteration":    iteration,
            "query":        query,
            "confidence":   round(confidence, 3),
            "chunks_found": len(chunks),
            "status":       "accepted"
        })

        if confidence > best_confidence:
            best_confidence = confidence
            best_chunks     = chunks

    return best_chunks, best_confidence, iteration_log