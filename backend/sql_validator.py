import re
from schema_allowlist import get_allowed_tables, get_allowed_columns

# Keywords that must NEVER appear in agent-generated SQL
FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "MERGE", "GRANT", "REVOKE", "REPLACE",
    "EXEC", "EXECUTE", "CALL", "LOCK", "UNLOCK", ";--", "/*"
]


class SQLValidationError(Exception):
    """Raised when generated SQL fails a security check."""
    pass


def validate_sql(sql: str) -> str:
    """
    Runs every security check against the generated SQL.
    Returns the cleaned SQL if it passes.
    Raises SQLValidationError if anything looks unsafe.
    """

    cleaned = sql.strip()
    upper   = cleaned.upper()

   # ── Check 1: must start with SELECT ───────────────────
    if not upper.startswith("SELECT"):
        raise SQLValidationError(
            "Only SELECT statements are allowed. Generated query rejected."
        )

  
    if re.search(r'SELECT\s+\*', upper):
        raise SQLValidationError(
            "SELECT * is not allowed. Queries must explicitly name columns "
            "so they can be checked against the allowed schema."
        )

    # ── Check 2: no forbidden keywords anywhere ───────────
    for keyword in FORBIDDEN_KEYWORDS:
        # Use word boundaries so "UPDATED_AT" doesn't false-positive on "UPDATE"
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, upper):
            raise SQLValidationError(
                f"Forbidden keyword detected: '{keyword}'. Query rejected."
            )

    # ── Check 3: no multiple statements (semicolon stacking) ──
   
    body = cleaned.rstrip(";").strip()
    if ";" in body:
        raise SQLValidationError(
            "Multiple SQL statements detected. Query rejected."
        )

    # ── Check 4: only allowlisted tables referenced ───────
    allowed_tables = get_allowed_tables()
    
    referenced_tables = re.findall(
        r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        cleaned,
        re.IGNORECASE
    )
    for table in referenced_tables:
        if table.lower() not in {t.lower() for t in allowed_tables}:
            raise SQLValidationError(
                f"Table '{table}' is not in the allowed schema. Query rejected."
            )

  # ── Check 5: only allowlisted columns referenced ──────
    all_allowed_columns = set()
    for table in allowed_tables:
        all_allowed_columns.update(c.lower() for c in get_allowed_columns(table))

    qualified_refs = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\.([a-zA-Z_][a-zA-Z0-9_]*)\b', cleaned)
    for col in qualified_refs:
        if col.lower() not in all_allowed_columns:
            raise SQLValidationError(
                f"Column '{col}' is not in the allowed schema. Query rejected."
            )

   
    sql_keywords = {
        "select", "from", "where", "join", "on", "and", "or", "limit",
        "order", "by", "group", "as", "asc", "desc", "in", "not", "null",
        "like", "between", "count", "sum", "avg", "max", "min", "distinct",
        "inner", "left", "right", "outer", "having", "is"
    }
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', cleaned)


    aliases_and_tables = {tb.lower() for tb in allowed_tables}
    aliases_and_tables.update(
        m.group(1).lower()
        for m in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.', cleaned)
    )

    for t in tokens:
        t_lower = t.lower()
        if (t_lower in sql_keywords
                or t_lower in aliases_and_tables
                or t.isdigit()):
            continue
       
        if t_lower not in all_allowed_columns:
           
            context_pattern = r'(?:SELECT|,|WHERE|AND|OR)\s+' + re.escape(t) + r'\b'
            if re.search(context_pattern, cleaned, re.IGNORECASE):
                raise SQLValidationError(
                    f"Column '{t}' is not in the allowed schema. Query rejected."
                )
    # ── Check 6: must include a LIMIT clause ──────────────
    # Exception: a pure aggregate query (COUNT/SUM/AVG/MAX/MIN)
   
    is_pure_aggregate = (
        re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', upper) is not None
        and "GROUP BY" not in upper
    )

    if "LIMIT" not in upper and not is_pure_aggregate:
        raise SQLValidationError(
            "Query must include a LIMIT clause. Query rejected."
        )

    return body