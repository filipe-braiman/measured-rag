def route_retrieval_query(
    rewritten_query,
    retrieval_mode="auto",
):
    """
    Resolve the retrieval strategy and its dense/BM25 fusion weight.

    Manual modes directly select a predefined fusion strategy.
    Auto mode analyzes the rewritten query and chooses a strategy using lightweight routing heuristics.
    """

    retrieval_mode = (
        retrieval_mode or "auto"
    ).strip().lower()

    # Manual retrieval modes
    manual_routes = {
        "keyword": 0.45,
        "semantic": 0.80,
        "balanced": 0.65,
    }

    if retrieval_mode in manual_routes:
        return (
            retrieval_mode,
            manual_routes[retrieval_mode],
        )

    # Auto mode
    q = rewritten_query.strip()
    q_lower = q.lower()
    tokens = q.split()

    keyword_signals = 0

    if len(tokens) <= 4:
        keyword_signals += 1

    if any(char.isdigit() for char in q):
        keyword_signals += 1

    if any(
        token.isupper() and len(token) >= 2
        for token in tokens
    ):
        keyword_signals += 1

    if '"' in q or "'" in q:
        keyword_signals += 1

    semantic_starters = (
        "what",
        "why",
        "how",
        "summarize",
        "explain",
        "describe",
        "compare",
        "discuss",
        "tell",
        "outline",
    )

    semantic_signal = any(
        q_lower.startswith(word + " ")
        or q_lower == word
        for word in semantic_starters
    )

    if keyword_signals >= 2 and not semantic_signal:
        return "keyword", 0.45

    if semantic_signal and keyword_signals == 0:
        return "semantic", 0.80

    return "balanced", 0.65