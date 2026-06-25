import os
import json
from groq import Groq
from dotenv import load_dotenv

from schema_allowlist import get_schema_description

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"


def is_in_scope(question: str) -> dict:
    """
    Decides whether a question can plausibly be answered using
    the allowed database schema, BEFORE any SQL is generated.

    Returns: {"in_scope": bool, "reason": str}
    """

    schema_text = get_schema_description()

    prompt = f"""You are a scope classifier for a business database assistant.

This database contains ONLY the following information:
{schema_text}

DECISION RULE — apply this mechanically, do not overthink:
If the question refers to any of the entities in the schema above
(employees, departments, projects) — using ANY phrasing such as "name",
"list", "show", "how many", "which", "who" — classify it as IN SCOPE.
You do NOT need a perfect match to a specific column. If the subject of
the question is one of the tables above, it is in scope. Only mark a
question OUT OF SCOPE if its subject is something entirely absent from
the schema (e.g. weather, geography, general trivia, coding help,
creative writing).

Examples of IN SCOPE questions:
- "Which employees work in Engineering?"
- "What projects is the Sales department running?"
- "How many departments do we have?"
- "Name 3 employees"
- "Name the active projects"
- "List the employees"
- "Who works in Sales?"

Examples of OUT OF SCOPE questions:
- "What's the capital of France?"
- "Write me a poem"
- "What's the weather today?"
- "Who won the World Cup in 2022?"

User question: {question}

Return ONLY valid JSON in this exact format, nothing else:
{{"in_scope": true or false, "reason": "<one short sentence explaining why>"}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0
        )
        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        return {
            "in_scope": bool(result.get("in_scope", False)),
            "reason":   result.get("reason", "")
        }

    except Exception as e:
        # Fail safe: if classification itself fails, treat as out of scope
        # rather than risk letting an unclassified question through.
        return {
            "in_scope": False,
            "reason": f"Could not classify scope, defaulting to out-of-scope for safety: {str(e)}"
        }