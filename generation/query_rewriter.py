from core.llm import get_llm
from core.models import rewriter_request_settings

def rewrite_query(user_query, chat_history):
    history_text = "\n".join(
        f"{msg['role'].title()}: {msg['content']}"
        for msg in chat_history[-4:]
    )

    prompt = f"""
You are a query rewriting assistant for a RAG system.

Max Reasoning Length: 100 tokens (DO NOT EXCEED THIS LIMIT!)

Your task is to rewrite the user's question into a single, concise query optimized for document retrieval.

Rules:
- Apply only minimal necessary rewriting to make the query retrieval-friendly
- Preserve the original meaning exactly
- Resolve vague references using the conversation history if needed (ONLY IF NEEDED)
- Prioritize keywords and the original semantic phrasing of the question
- Make the query self-contained and explicit
- Do not answer the question
- Do not add extra details not present in the conversation
- Return only the rewritten query, nothing else

Conversation history:
{history_text}

User question:
{user_query}

Rewritten query:
""".strip()

    try:
        request = rewriter_request_settings()
        response = get_llm().chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **request,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else user_query
    except Exception:
        return user_query