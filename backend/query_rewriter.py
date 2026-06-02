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


def rewrite_query(question):

    prompt = f"""
Rewrite the following user question into a clear,
search-optimized query for retrieving relevant
document chunks.

Question:
{question}

Only return the rewritten query.
"""

    response = model.generate_content(prompt)

    return response.text.strip()