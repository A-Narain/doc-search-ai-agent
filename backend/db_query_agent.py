from scope_classifier import is_in_scope
from sql_generator import generate_sql
from sql_executor import execute_safe_query


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
        return {
            "success":   True,
            "stage":     "complete",
            "sql":       generated_sql,
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