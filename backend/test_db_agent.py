from db_query_agent import run_database_query

questions = [
    "List all active projects and which department runs them",
    "What's the capital of France?",
    "Which employees work in Engineering?",
    "Write me a haiku about autumn",
]

for q in questions:
    print(f"\n=== {q} ===")
    result = run_database_query(q)
    print(result)