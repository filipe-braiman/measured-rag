def distance_to_similarity(distance: float) -> float:
    return 1 / (1 + distance)

def truncate_text(text, max_chars=1000):
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"

def prepare_submission(user_query, chat_history):
    user_query = (user_query or "").strip()

    # Empty/whitespace-only message: no-op
    if not user_query:
        return (
            list(chat_history or []),  # chatbot unchanged
            "",                        # pending query stays empty
            "",                        # textbox remains empty
        )

    from deployment.guardrails import query_rejection
    rejection = query_rejection(user_query)
    if rejection:
        import gradio as gr
        gr.Warning(rejection)
        return list(chat_history or []), "", ""

    display_history = list(chat_history or [])

    display_history.extend(
        [
            {
                "role": "user",
                "content": user_query,
            },
            {
                "role": "assistant",
                "content": "Processing…",
            },
        ]
    )

    return (
        display_history,  # temporary chatbot view
        user_query,       # preserve submitted text
        "",               # clear typing box immediately
    )

def prepare_session_submission(user_query, chat_history, query_count):
    """Validate first, then consume session allowance before downstream work."""
    from deployment.guardrails import consume_query_allowance, QueryLimitError

    display, pending, textbox = prepare_submission(user_query, chat_history)
    if not pending:
        return display, pending, textbox, query_count
    try:
        next_count = consume_query_allowance(query_count)
    except QueryLimitError:
        import gradio as gr
        # Only cloud mode raises this error. Success-only event wiring prevents
        # stale pending text from reaching chat; the existing state is retained.
        raise gr.Error("This public demo allows up to 5 questions per session.",
                       print_exception=False) from None
    return display, pending, textbox, next_count
