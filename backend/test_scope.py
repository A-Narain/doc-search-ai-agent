from scope_classifier import is_in_scope

questions = [
    "Which employees work in Engineering?",
    "What's the capital of France?",
    "List all active projects",
    "Write me a haiku about autumn",
    "How many departments do we have?",
    "What's today's weather in Mumbai?",
    "Name 3 employees",
    "Name the active projects",
]

for q in questions:
    result = is_in_scope(q)
    status = "IN SCOPE" if result["in_scope"] else "OUT OF SCOPE"
    print(f"\n[{status}] {q}")
    print(f"  Reason: {result['reason']}")