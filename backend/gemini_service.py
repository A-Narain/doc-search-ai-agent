import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"


def generate_answer(question, retrieved_chunks):

    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])

    prompt = f"""You are a document search assistant.
Answer the question using ONLY the context below.
If the context doesn't contain enough information, say so clearly.
Be concise and factual.

Context:
{context}

Question:
{question}

Answer:"""

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
                print(f"[Groq] Error, retrying in 3s... {e}")
                time.sleep(3)
            else:
                raise