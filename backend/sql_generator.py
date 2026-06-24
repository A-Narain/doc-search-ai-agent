import os
from groq import Groq
from dotenv import load_dotenv

from schema_allowlist import get_schema_description

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"


def generate_sql(question: str) -> str:
    schema_text = get_schema_description()

    prompt = f"""You are a MySQL query generator.

Database schema (these are the ONLY tables and columns that exist):
{schema_text}

Rules:
- Generate exactly ONE SQL SELECT statement and nothing else
- Only use tables and columns listed in the schema above
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, MERGE, GRANT, REVOKE
- Always include a LIMIT clause, maximum 50 rows
- Return ONLY the raw SQL. No explanation, no markdown, no semicolon at the end

User question: {question}

SQL:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0
    )

    sql = response.choices[0].message.content.strip()

    # Strip markdown code fences if the LLM adds them anyway
    if sql.startswith("```"):
        sql = sql.split("```")[1]
        if sql.lower().startswith("sql"):
            sql = sql[3:]
    sql = sql.strip().rstrip(";")

    return sql