import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"


def rewrite_query(question):

    prompt = f"""Rewrite the following user question into a clear, concise search query optimized for semantic document retrieval.
Use specific keywords. Remove filler words.
Return ONLY the rewritten query. No explanation, no quotes.

Question: {question}

Rewritten query:"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                print(f"[Groq] Error, retrying in 3s... {e}")
                time.sleep(3)
            else:
                return question  # fallback to original question