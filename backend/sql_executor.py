from db_connection import get_readonly_connection
from sql_validator import validate_sql, SQLValidationError


class SQLExecutionError(Exception):
    """Raised when a validated query fails during execution."""
    pass


def execute_safe_query(sql: str) -> dict:
    """
    Validates and executes a SQL query against the read-only connection.

    Returns a dict with either:
      - {"success": True, "columns": [...], "rows": [...], "row_count": N}
      - {"success": False, "error": "..."}
    """

    # ── Always validate first, no exceptions ──────────────
    try:
        safe_sql = validate_sql(sql)
    except SQLValidationError as e:
        return {
            "success": False,
            "error": f"Query rejected by security validator: {str(e)}"
        }

    # ── Execute against the read-only connection ──────────
    conn = None
    try:
        conn = get_readonly_connection()
        cursor = conn.cursor()
        cursor.execute(safe_sql)

        columns = [desc[0] for desc in cursor.description]
        rows    = cursor.fetchall()

        return {
            "success":   True,
            "columns":   columns,
            "rows":      [list(row) for row in rows],
            "row_count": len(rows)
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Database execution error: {str(e)}"
        }

    finally:
        if conn is not None:
            conn.close()