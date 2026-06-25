from scope_classifier import is_in_scope
from sql_generator import generate_sql
from sql_executor import execute_safe_query


import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"


def format_db_answer(question: str, columns: list, rows: list) -> str:
    """
    Turns raw SQL result rows into a natural language answer.
    """
    if not rows:
        return "I queried the database, but no matching records were found."

    # Build a readable table block to give the LLM as context
    table_lines = [", ".join(columns)]
    for row in rows:
        table_lines.append(", ".join(str(v) for v in row))
    table_text = "\n".join(table_lines)

    prompt = f"""You are a helpful assistant answering a question using database query results.

Question: {question}

Query results (CSV format):
{table_text}

Write a clear, natural language answer to the question using ONLY this data.
Do not invent any information not present in the results.
If there are multiple results, you may use a short list.
Return ONLY the answer text, no preamble.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # Fallback to the raw table if the LLM call fails
        return f"Found {len(rows)} result(s):\n\n{table_text}"


def run_database_query(question: str) -> dict:
    """
    Full pipeline: scope check → generate SQL → validate → execute.

    If the question is out of scope, SQL is never generated at all —
    this is an explicit refusal, not a fallback to general knowledge.
    """

    # ── Step 0: scope gate — runs before anything else ────
    scope_result = is_in_scope(question)

    if not scope_result["in_scope"]:
        return {
            "success": False,
            "stage":   "scope_check",
            "sql":     None,
            "answer":  (
                "I can only answer questions about the data in this "
                "application's database. This question is outside that scope, "
                "so I'm not able to help with it here."
            ),
            "reason":  scope_result["reason"]
        }

    # ── Step 1: generate SQL from the question ────────────
    try:
        generated_sql = generate_sql(question)
    except Exception as e:
        return {
            "success": False,
            "stage":   "generation",
            "sql":     None,
            "answer":  "I couldn't generate a database query for this question.",
            "error":   str(e)
        }

    # ── Step 2 + 3: validate and execute ──────────────────
    result = execute_safe_query(generated_sql)

    if result["success"]:
        answer = format_db_answer(question, result["columns"], result["rows"])
        return {
            "success":   True,
            "stage":     "complete",
            "sql":       generated_sql,
            "answer":    answer,
            "columns":   result["columns"],
            "rows":      result["rows"],
            "row_count": result["row_count"]
        }
    else:
        return {
            "success": False,
            "stage":   "validation_or_execution",
            "sql":     generated_sql,
            "answer":  "I generated a query for this, but it didn't pass security checks, so I won't run it.",
            "error":   result["error"]
        }