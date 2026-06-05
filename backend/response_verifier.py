import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"


def verify_answer(question, answer, retrieved_chunks):

    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])

    # ── Step 1: Verify ────────────────────────────────────
    verify_prompt = f"""You are a strict fact-checker for a document search system.

Question: {question}

Retrieved context (source of truth):
{context}

Generated answer:
{answer}

Does the answer contain ONLY information supported by the context above?
Reply with exactly one word: VERIFIED or NOT VERIFIED"""

    status = "VERIFIED"

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": verify_prompt}],
                max_tokens=10,
                temperature=0.0
            )
            raw = response.choices[0].message.content.strip().upper()
            status = "NOT VERIFIED" if "NOT" in raw else "VERIFIED"
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(3)

    # ── Step 2: Refine if not verified ────────────────────
    final_answer = answer

    if status == "NOT VERIFIED":
        refine_prompt = f"""You are a response refiner.

The answer below contains claims not supported by the context.
Rewrite it using ONLY information from the context.
If the context is insufficient, say: "The documents do not contain enough information to answer this fully."

Question: {question}

Context (only source of truth):
{context}

Answer to refine:
{answer}

Refined answer:"""

        for attempt in range(3):
            try:
                refine_response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": refine_prompt}],
                    max_tokens=1000,
                    temperature=0.1
                )
                final_answer = refine_response.choices[0].message.content.strip()
                print("[Verifier] Answer was NOT VERIFIED — refined.")
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(3)

    return status, final_answer