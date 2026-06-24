from sql_validator import validate_sql, SQLValidationError


def run_test(label, sql):
    print(f"\n--- {label} ---")
    print(f"SQL: {sql}")
    try:
        result = validate_sql(sql)
        print(f"PASSED: {result}")
    except SQLValidationError as e:
        print(f"REJECTED: {e}")


# ── Test 1: a normal, valid query ──────────────────────────
run_test(
    "Valid query",
    "SELECT e.first_name, e.last_name FROM employees e "
    "JOIN departments d ON e.department_id = d.department_id "
    "WHERE d.department_name = 'Engineering' LIMIT 50"
)

# ── Test 2: a destructive query ────────────────────────────
run_test(
    "Destructive query (should be rejected)",
    "DELETE FROM employees WHERE employee_id = 1"
)

# ── Test 3: trying to reach a sensitive column ─────────────
run_test(
    "Sensitive column query (should be rejected)",
    "SELECT first_name, salary FROM employees LIMIT 10"
)

# ── Test 4: trying to reach a non-allowlisted table ────────
run_test(
    "Non-allowlisted table (should be rejected)",
    "SELECT * FROM users LIMIT 10"
)

# ── Test 5: missing LIMIT clause ───────────────────────────
run_test(
    "Missing LIMIT (should be rejected)",
    "SELECT first_name, last_name FROM employees"
)

run_test(
    "SELECT * (should be rejected)",
    "SELECT * FROM employees LIMIT 10"
)