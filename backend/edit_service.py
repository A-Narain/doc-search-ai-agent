import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"


def apply_edit_to_chunk(original_chunk: str, edit_instruction: str) -> str:
    prompt = f"""You are a precise document editor.

Edit instruction:
{edit_instruction}

Original text:
{original_chunk}

Rewrite ONLY the original text above, applying the edit instruction exactly.
- Keep all unrelated content unchanged word for word.
- Do not add explanations, preamble, or commentary.
- Return ONLY the rewritten text.
"""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                raise


def identify_chunks_to_edit(edit_instruction: str, retrieved_chunks: list) -> list:
    chunks_text = "\n\n".join([
        f"[Chunk {i}] (file: {c['filename']}, id: {c['chunk_id']}):\n{c['text']}"
        for i, c in enumerate(retrieved_chunks)
    ])

    prompt = f"""You are a document editing assistant.

Edit instruction: {edit_instruction}

Retrieved document chunks:
{chunks_text}

Which chunk numbers (0-indexed) contain the content that needs to be edited?
Return ONLY a comma-separated list of chunk numbers. Example: 0,2
If none need editing, return: none
"""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            raw = response.choices[0].message.content.strip().lower()

            if raw == "none":
                return []

            indices = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
            return [retrieved_chunks[i] for i in indices if i < len(retrieved_chunks)]
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                return retrieved_chunks


def rebuild_document_with_edits(original_text: str, edited_chunks: list[dict]) -> str:
    updated_text = original_text
    for chunk in edited_chunks:
        old_text = chunk["original_text"]
        new_text = chunk["rewritten_text"]
        if old_text in updated_text:
            updated_text = updated_text.replace(old_text, new_text, 1)
        else:
            updated_text += f"\n\n[EDITED SECTION]:\n{new_text}"
    return updated_text