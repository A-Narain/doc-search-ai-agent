import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"


def format_response(
    question,
    answer,
    retrieved_chunks,
    confidence,
    verification,
    iteration_log
):

    # ── Confidence label ──────────────────────────────────
    if confidence >= 0.60:
        confidence_label = "High"
        confidence_note  = "Strong match found in documents."
    elif confidence >= 0.35:
        confidence_label = "Medium"
        confidence_note  = "Partial match. Answer may be incomplete."
    else:
        confidence_label = "Low"
        confidence_note  = "Weak match. Answer may not fully reflect document content."

    # ── Format with inline citations ──────────────────────
    sources_text = "\n".join([
        f"[{i+1}] {c['filename']} (score {c['score']}): {c['text'][:120]}..."
        for i, c in enumerate(retrieved_chunks)
    ])

    format_prompt = f"""You are a response formatter for a document search system.

Question: {question}

Raw answer: {answer}

Source chunks:
{sources_text}

Rewrite the answer as clear professional prose.
Add inline citations like [1], [2] where facts come from specific sources.
If there are multiple distinct points, use a short numbered list.
End with: Sources: [1] filename, [2] filename (deduplicated).
Return ONLY the formatted answer. No preamble."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": format_prompt}],
            max_tokens=1200,
            temperature=0.1
        )
        formatted_answer = response.choices[0].message.content.strip()
    except Exception:
        formatted_answer = answer  # fallback

    # ── Agent trace summary ───────────────────────────────
    total_iterations = len(iteration_log)
    final_status     = iteration_log[-1]["status"] if iteration_log else "unknown"

    agent_summary = (
        f"Query resolved in {total_iterations} iteration(s). "
        f"Final status: {final_status}."
    )
    if total_iterations > 1:
        agent_summary += (
            f" Agent refined the search {total_iterations - 1} time(s) "
            f"to improve retrieval quality."
        )

    # ── Deduplicated source list ──────────────────────────
    sources = []
    seen    = set()
    for chunk in retrieved_chunks:
        if chunk["filename"] not in seen:
            seen.add(chunk["filename"])
            sources.append({
                "filename": chunk["filename"],
                "score":    chunk["score"],
                "chunk_id": chunk["chunk_id"]
            })

    # ── Disclaimer if not verified ────────────────────────
    disclaimer = None
    if verification == "NOT VERIFIED":
        disclaimer = (
            "This answer could not be fully verified against source documents. "
            "It has been refined to only include supported claims."
        )

    return {
        "question":         question,
        "answer":           formatted_answer,
        "confidence_score": round(confidence, 3),
        "confidence_label": confidence_label,
        "confidence_note":  confidence_note,
        "verification":     verification,
        "disclaimer":       disclaimer,
        "agent_summary":    agent_summary,
        "agent_iterations": iteration_log,
        "sources":          sources
    }