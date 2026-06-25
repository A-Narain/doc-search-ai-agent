from intent_classifier import classify_intent
from agent_router import route

test_messages = [
    "Which employees work in Engineering?",
    "What's the capital of France?",
    "List all active projects",
    "name the active project",
    "how many employees are there?",
]

for msg in test_messages:
    classified = classify_intent(msg, "No prior conversation.", [], scoped_file=None)
    print(f"\n=== {msg} ===")
    print(f"Classified intent: {classified['intent']}")
    result = route(classified, msg, "No prior conversation.")
    print(result)