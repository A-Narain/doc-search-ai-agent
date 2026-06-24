import os
import time
from groq import Groq
from dotenv import load_dotenv

from retriever import retrieve_chunks
from query_rewriter import rewrite_query

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"

MAX_ITERATIONS       = 3
CONFIDENCE_THRESHOLD = 0.60   # "good enough to answer confidently"
MINIMUM_FLOOR        = 0.15   # below this, there's genuinely not enough info


def compute_confidence(retrieved_chunks):
    if not retrieved_chunks:
        return 0.0
    scores = [chunk["score"] for chunk in retrieved_chunks]
    return sum(scores) / len(scores)


def generate_alternative_query(original_question, current_query, confidence):

    prompt = f"""A document search query returned poor results.

Original question: {original_question}
Current query that failed: {current_query}
Confidence score: {confidence:.2f} out of 1.0

Generate ONE alternative search query using different keywords, synonyms, or a different angle.
Return ONLY the new query. No explanation."""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                return current_query  # fallback


def agentic_retrieve(question, filename_filter=None, session_id=None):

    query           = rewrite_query(question)
    best_chunks     = []
    best_confidence = 0.0
    iteration_log   = []

    for iteration in range(1, MAX_ITERATIONS + 1):

        print(f"\n[Agent] Iteration {iteration}/{MAX_ITERATIONS}")
        print(f"[Agent] Query: {query}")
        if session_id:
            print(f"[Agent] Session: {session_id}")

        chunks = retrieve_chunks(
            query,
            filename_filter=filename_filter,
            session_id=session_id
        )

        confidence = compute_confidence(chunks)

        print(f"[Agent] Confidence: {confidence:.3f} (threshold: {CONFIDENCE_THRESHOLD})")

        iteration_log.append({
            "iteration":    iteration,
            "query":        query,
            "confidence":   round(confidence, 3),
            "chunks_found": len(chunks),
            "status":       "accepted"               if confidence >= CONFIDENCE_THRESHOLD
                            else "retrying"           if iteration < MAX_ITERATIONS
                            else "max_iterations_reached"
        })

        if confidence > best_confidence:
            best_confidence = confidence
            best_chunks     = chunks

        if confidence >= CONFIDENCE_THRESHOLD:
            print(f"[Agent] Confident enough. Stopping at iteration {iteration}.")
            break

        if iteration < MAX_ITERATIONS:
            print(f"[Agent] Low confidence. Generating alternative query...")
            query = generate_alternative_query(question, query, confidence)
        else:
            print(f"[Agent] Max iterations reached. Returning best result.")

    # ── Adequacy gate ──────────────────────────────────────
    # Explicit yes/no: is there enough information to answer
    # accurately, or should the agent refuse rather than risk
    # fabricating an answer from weak/irrelevant chunks?
    is_adequate = best_confidence >= MINIMUM_FLOOR and len(best_chunks) > 0

    return best_chunks, best_confidence, iteration_log, is_adequate