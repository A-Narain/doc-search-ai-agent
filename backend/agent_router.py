import os
import time
from groq import Groq
from dotenv import load_dotenv

from agent_loop import agentic_retrieve
from gemini_service import generate_answer
from response_verifier import verify_answer
from response_formatter import format_response
from edit_service import (
    identify_chunks_to_edit,
    apply_edit_to_chunk,
    rebuild_document_with_edits
)

from db_query_agent import run_database_query
from chunking import chunk_text
from vector_store import store_chunks, collection
from github_service import upload_to_github
from document_processor import extract_text

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

UPLOAD_FOLDER = "uploads"


def groq_generate(prompt: str, max_tokens: int = 1000) -> str:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                print(f"[Groq] Error, retrying in 3s... {e}")
                time.sleep(3)
            else:
                raise


# ── Tool: Answer a question ───────────────────────────────

def handle_question(refined_message, original_message, target_files, conversation_history):
    filename_filter = target_files[0] if len(target_files) == 1 else None

    retrieved_chunks, confidence, iteration_log, is_adequate = agentic_retrieve(
        refined_message,
        filename_filter=filename_filter
    )

    if not retrieved_chunks:
        print("[Router] No chunks retrieved — declining, no general knowledge fallback")
        return {
        "intent":  "question",
        "answer":  "I couldn't find anything relevant in your uploaded documents to answer this.",
        "sources": []
    }

    # ── Adequacy gate ───────────────────────────────────────
    # If retrieval confidence never cleared the minimum floor,
    # refuse rather than risk generating a fabricated answer.
    if not is_adequate:
        print(f"[Router] Retrieval inadequate (confidence: {confidence:.3f}) — refusing to fabricate")
        return {
            "intent":  "question",
            "answer":  (
                "I don't have enough relevant information in the indexed documents "
                "to answer this accurately. Try rephrasing your question, scoping to "
                "a specific document, or uploading a file that covers this topic."
            ),
            "confidence":       round(confidence, 3),
            "verification":     "NOT_ATTEMPTED",
            "agent_iterations": iteration_log,
            "sources":          []
        }

    # Guard: if multiple docs returned but none was specifically requested, ask user to clarify
    if not filename_filter:
        docs_found = {c["filename"] for c in retrieved_chunks}
        if len(docs_found) > 1:
            filenames = ", ".join(sorted(docs_found))
            return {
                "intent": "question",
                "answer": f"I found relevant content across multiple documents ({filenames}). Could you clarify which one you're referring to?",
                "sources": list(docs_found)
            }

    contextual_question = (
        f"Conversation so far:\n{conversation_history}\n\nCurrent question: {original_message}"
    ) if conversation_history != "No prior conversation." else original_message

    answer = generate_answer(contextual_question, retrieved_chunks)
    verification = verify_answer(original_message, answer, retrieved_chunks)

    return format_response(
        question=original_message,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        confidence=confidence,
        verification=verification,
        iteration_log=iteration_log
    )


# ── Tool: Summarise one or more documents ────────────────

def handle_summarise(target_files, conversation_history):
    all_chunks = []

    for filename in target_files:
        chunks, _, _, _ = agentic_retrieve(
            f"summarise {filename}",
            filename_filter=filename
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        return {
            "intent":  "summarise",
            "answer":  "No content found for the requested document(s).",
            "sources": []
        }

    files_label = ", ".join(target_files) if target_files else "all documents"

    prompt = f"""Provide a clear, structured summary of the following document content.
Organise by key themes. Be concise but comprehensive.

Document(s): {files_label}

Content:
{chr(10).join(c['text'] for c in all_chunks)}
"""
    answer = groq_generate(prompt)
    sources = list({c["filename"] for c in all_chunks})

    return {
        "intent":  "summarise",
        "answer":  answer,
        "sources": sources
    }


# ── Tool: Compare across documents ───────────────────────

def handle_compare(refined_message, target_files):
    if len(target_files) < 2:
        return {
            "intent": "compare",
            "answer": "Please specify at least two documents to compare.",
            "sources": []
        }

    per_file_chunks = {}
    for filename in target_files:
        chunks, _, _, _ = agentic_retrieve(refined_message, filename_filter=filename)
        per_file_chunks[filename] = chunks

    sections = []
    for filename, chunks in per_file_chunks.items():
        text = "\n".join(c["text"] for c in chunks)
        sections.append(f"--- {filename} ---\n{text}")

    prompt = f"""Compare the following document sections on the topic: "{refined_message}"

{chr(10).join(sections)}

Structure your comparison with:
1. Key similarities
2. Key differences
3. Summary verdict

Be specific and cite which document says what.
"""
    answer = groq_generate(prompt)

    return {
        "intent":  "compare",
        "answer":  answer,
        "sources": target_files
    }


# ── Tool: Edit a document ────────────────────────────────

def handle_edit(edit_instruction, target_files):
    if not target_files:
        return {
            "intent": "edit",
            "answer": "Please specify which document you want to edit. Use the scope dropdown to select a file, or mention the filename.",
            "sources": []
        }

    filename = target_files[0]
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        return {
            "intent": "edit",
            "answer": f"File '{filename}' not found on disk.",
            "sources": []
        }

    retrieved_chunks, _, iteration_log, _ = agentic_retrieve(
        edit_instruction,
        filename_filter=filename
    )

    if not retrieved_chunks:
        return {
            "intent": "edit",
            "answer": f"No relevant content found in '{filename}' for that instruction.",
            "sources": []
        }

    chunks_to_edit = identify_chunks_to_edit(edit_instruction, retrieved_chunks)

    if not chunks_to_edit:
        return {
            "intent": "edit",
            "answer": "I couldn't identify which section to edit. Please be more specific.",
            "sources": []
        }

    edited = []
    for chunk in chunks_to_edit:
        rewritten = apply_edit_to_chunk(chunk["text"], edit_instruction)
        edited.append({
            "filename":       chunk["filename"],
            "chunk_id":       chunk["chunk_id"],
            "original_text":  chunk["text"],
            "rewritten_text": rewritten
        })

    original_text = extract_text(filepath)
    updated_text  = rebuild_document_with_edits(original_text, edited)

    result = collection.get(where={"filename": filename}, include=["metadatas"])
    old_ids = result.get("ids", [])
    if old_ids:
        collection.delete(ids=old_ids)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(updated_text)

    new_chunks = chunk_text(updated_text)
    store_chunks(new_chunks, filename)

    try:
        upload_to_github(filepath, filename)
    except Exception as e:
        print(f"[GitHub] Edit push failed: {e}")

    changes_summary = "\n".join([
        f"- Chunk {e['chunk_id']}: \"{e['original_text'][:80]}...\" → \"{e['rewritten_text'][:80]}...\""
        for e in edited
    ])

    return {
        "intent":        "edit",
        "answer":        f"Done. Made {len(edited)} edit(s) to '{filename}':\n\n{changes_summary}",
        "filename":      filename,
        "edits_made":    len(edited),
        "chunks_stored": len(new_chunks),
        "sources":       [filename]
    }


# ── Tool: List documents ──────────────────────────────────

def handle_list():
    result = collection.get(include=["metadatas"])
    metadatas = result.get("metadatas", [])

    seen = {}
    for meta in metadatas:
        fname = meta["filename"]
        seen[fname] = seen.get(fname, 0) + 1

    if not seen:
        return {
            "intent": "list",
            "answer": "No documents are currently indexed. Please upload some files first.",
            "sources": []
        }

    doc_lines = "\n".join(
        f"- {fname} ({count} chunks)"
        for fname, count in sorted(seen.items())
    )

    return {
        "intent":  "list",
        "answer":  f"I have the following {len(seen)} document(s) indexed:\n\n{doc_lines}",
        "sources": list(seen.keys())
    }


# ── Tool: Chitchat ────────────────────────────────────────

def handle_chitchat(message, conversation_history):
    prompt = f"""You are a helpful AI document assistant.
The user said something that isn't a document query.
Respond helpfully and briefly, and gently guide them back to document-related tasks if appropriate.

Conversation so far:
{conversation_history}

User: {message}
"""
    answer = groq_generate(prompt, max_tokens=300)
    return {
        "intent":  "chitchat",
        "answer":  answer,
        "sources": []
    }

# ── Tool: Database query ──────────────────────────────────

def handle_database_query(original_message):
    result = run_database_query(original_message)

    if not result["success"]:
        # Covers both out-of-scope refusals and validation/execution failures
        return {
            "intent":  "database_query",
            "answer":  result.get("answer", "I couldn't process that database question."),
            "sql":     result.get("sql"),
            "sources": []
        }

    # ── Format the raw rows into a readable answer ────────
    columns   = result["columns"]
    rows      = result["rows"]
    row_count = result["row_count"]

    if row_count == 0:
        answer = "I ran a query against the database, but no matching records were found."
    else:
        lines = [", ".join(columns)]
        for row in rows:
            lines.append(", ".join(str(v) for v in row))
        answer = (
            f"Found {row_count} result(s):\n\n" + "\n".join(lines)
        )

    return {
        "intent":    "database_query",
        "answer":    answer,
        "sql":       result["sql"],
        "row_count": row_count,
        "sources":   []
    }





# ── Main router ───────────────────────────────────────────

def route(classified_intent: dict, original_message: str, conversation_history: str) -> dict:
    intent           = classified_intent["intent"]
    refined_message  = classified_intent["refined_message"]
    target_files     = classified_intent["target_files"]
    edit_instruction = classified_intent.get("edit_instruction")

    print(f"[Router] Intent: {intent} | Files: {target_files} | Confidence: {classified_intent['confidence']}")

    if intent in ("question", "clarify"):
        return handle_question(refined_message, original_message, target_files, conversation_history)
    elif intent == "summarise":
        return handle_summarise(target_files, conversation_history)
    elif intent == "compare":
        return handle_compare(refined_message, target_files)
    elif intent == "edit":
        return handle_edit(edit_instruction or refined_message, target_files)
    elif intent == "list":
        return handle_list()
    elif intent == "chitchat":
        return handle_chitchat(original_message, conversation_history)
    elif intent == "general_knowledge":
       return {
        "intent":  "general_knowledge",
        "answer":  "I'm only able to help with questions about your uploaded documents. I can't answer general knowledge questions unrelated to them.",
        "sources": []
              }
    elif intent == "database_query":
        return handle_database_query(original_message)
    else:
        return handle_question(refined_message, original_message, target_files, conversation_history)