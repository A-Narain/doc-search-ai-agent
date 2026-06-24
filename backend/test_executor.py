from sql_executor import execute_safe_query


def run_test(label, sql):
    print(f"\n--- {label} ---")
    print(f"SQL: {sql}")
    result = execute_safe_query(sql)
    if result["success"]:
        print(f"Columns: {result['columns']}")
        print(f"Rows ({result['row_count']}):")
        for row in result["rows"]:
            print(f"  {row}")
    else:
        print(f"FAILED: {result['error']}")


# ── A real, valid query ────────────────────────────────────
run_test(
    "Valid query — should return real data",
    "SELECT e.first_name, e.last_name, d.department_name "
    "FROM employees e "
    "JOIN departments d ON e.department_id = d.department_id "
    "WHERE d.department_name = 'Engineering' LIMIT 50"
)

# ── A destructive query — should be blocked before reaching MySQL ──
run_test(
    "Destructive query — should be blocked",
    "DELETE FROM employees WHERE employee_id = 1"
)

# ── A sensitive column query — should be blocked ──────────
run_test(
    "Sensitive column — should be blocked",
    "SELECT first_name, ssn FROM employees LIMIT 10"
)