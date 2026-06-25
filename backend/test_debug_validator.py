from sql_generator import generate_sql
from sql_validator import validate_sql, SQLValidationError

questions = [
    "How many employees are there?",
    "How many more departments do you have?",
    "Name 3 employees",
]

for q in questions:
    print(f"\n=== {q} ===")
    sql = generate_sql(q)
    print(f"Generated SQL: {repr(sql)}")
    try:
        result = validate_sql(sql)
        print(f"PASSED: {result}")
    except SQLValidationError as e:
        print(f"REJECTED: {e}")