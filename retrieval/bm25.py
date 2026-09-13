def get_bm25_scores_filtered(
    kb_state,
    query,
    allowed_doc_ids,
    top_n,
):
    bm25_scores = {}

    bm25_index = kb_state.get("bm25_index")
    bm25_chunks = kb_state.get("bm25_chunks", [])

    if not bm25_index or not bm25_chunks:
        return bm25_scores

    top_n = max(1, int(top_n))

    tokenized_query = query.split()
    scores = bm25_index.get_scores(tokenized_query)

    allowed_results = []

    for i, score in enumerate(scores):
        doc = bm25_chunks[i]

        if doc.metadata.get("doc_id") not in allowed_doc_ids:
            continue

        allowed_results.append(
            (
                doc.metadata.get("chunk_id"),
                float(score),
            )
        )

    if not allowed_results:
        return bm25_scores

    # Highest raw BM25 scores first
    allowed_results.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    # Keep only BM25 Top-N candidates
    top_results = allowed_results[:top_n]

    raw_scores = [
        score
        for _, score in top_results
    ]

    min_score = min(raw_scores)
    max_score = max(raw_scores)

    if max_score == min_score:
        for chunk_id, _ in top_results:
            bm25_scores[chunk_id] = 1.0
    else:
        for chunk_id, score in top_results:
            bm25_scores[chunk_id] = (
                (score - min_score)
                / (max_score - min_score)
            )

    return bm25_scores