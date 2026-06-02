import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)


def verify_answer(
    question,
    answer,
    retrieved_chunks
):

    context = "\n\n".join(
        [
            chunk["text"]
            for chunk in retrieved_chunks
        ]
    )

    prompt = f"""
You are a response verifier.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Check whether the answer is fully supported by the retrieved context.

If supported:
Return exactly:
VERIFIED

If not supported:
Return exactly:
NOT VERIFIED
"""

    response = model.generate_content(
        prompt
    )

    return response.text.strip()