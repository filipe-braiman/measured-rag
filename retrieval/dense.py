def get_dense_results_filtered(
    vectordb,
    query,
    allowed_doc_ids,
    top_n,
):
    if vectordb is None:
        return []

    top_n = max(1, int(top_n))

    try:
        dense_results = vectordb.similarity_search_with_score(
            query,
            k=top_n,
        )
    except Exception:
        return []

    filtered = []

    for doc, dist in dense_results:
        if doc.metadata.get("doc_id") in allowed_doc_ids:
            filtered.append((doc, dist))

    return filtered