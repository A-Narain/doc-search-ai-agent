import os
import json
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

INTENTS = {
    "question":          "User is asking a question to be answered from documents",
    "edit":              "User wants to modify content inside a specific document",
    "summarise":         "User wants a summary of one or more documents",
    "compare":           "User wants to compare content across two or more documents",
    "list":              "User wants to know which documents are indexed",
    "delete":            "User wants to delete a document from the index",
    "clarify":           "User is following up or referring to something from earlier in the conversation",
    "chitchat":          "General conversation, greetings, small talk",
    "general_knowledge": "A factual or conceptual question not related to any uploaded document",
}


def classify_intent(user_message, conversation_history, available_files, scoped_file=None):
    files_str = ", ".join(available_files) if available_files else "none"
    intent_descriptions = "\n".join(f'  "{k}": {v}' for k, v in INTENTS.items())

    scope_note = (
        f'\nThe user has currently scoped the conversation to this specific file: "{scoped_file}". '
        f'If target_files is ambiguous or unmentioned, default to this file.\n'
        if scoped_file else ""
    )

    prompt = f"""You are an intent classifier for a document AI agent.

Available documents in the system:
{files_str}
{scope_note}
Recent conversation history:
{conversation_history}

Current user message:
{user_message}

Classify the user's intent and extract parameters. Return ONLY valid JSON with this exact structure:
{{
  "intent": "<one of: question, edit, summarise, compare, list, delete, clarify, chitchat, general_knowledge>",
  "refined_message": "<rewrite the user message resolving any references to prior conversation>",
  "target_files": ["<filename if mentioned or inferable, else empty list>"],
  "edit_instruction": "<only if intent is edit: the specific change to make, else null>",
  "confidence": <0.0 to 1.0>
}}

Intent definitions:
{intent_descriptions}

Rules:
- If the user uses words like "remove", "delete", "change", "update", "replace", "edit", "modify", "rewrite" anything INSIDE a document, classify as "edit"
- For "edit" intent, target_files MUST be filled — infer the filename from conversation history, the scoped file, or available documents if not explicitly stated
- If only one document is available and the user says "my resume", "the document", "it", "that file" — use that filename
- If a scoped file is provided above and the user's message doesn't name a different file, prefer the scoped file for target_files
- NEVER classify document modification requests as "question" or "general_knowledge"
- Only classify as "question" if the user is asking for information FROM a document, not changing it
- If the question is clearly about general knowledge and does NOT reference any uploaded document, classify as "general_knowledge"
- target_files should only contain filenames from the available documents list above
- For compare intent, target_files should contain 2+ files
- Return ONLY the JSON object, no markdown, no explanation
"""

    result = None

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            raw = response.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)
            break

        except json.JSONDecodeError:
            result = {
                "intent":           "question",
                "refined_message":  user_message,
                "target_files":     [scoped_file] if scoped_file else [],
                "edit_instruction": None,
                "confidence":       0.5
            }
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                result = {
                    "intent":           "question",
                    "refined_message":  user_message,
                    "target_files":     [scoped_file] if scoped_file else [],
                    "edit_instruction": None,
                    "confidence":       0.5
                }

    result.setdefault("intent",           "question")
    result.setdefault("refined_message",  user_message)
    result.setdefault("target_files",     [scoped_file] if scoped_file else [])
    result.setdefault("edit_instruction", None)
    result.setdefault("confidence",       0.5)

    return result