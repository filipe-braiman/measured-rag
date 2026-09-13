from utils.helpers import distance_to_similarity

def build_debug_info(
    original_query,
    rewritten_query,
    retrieval_route,
    fusion_alpha,
    raw_docs,
    reranked_docs,
    reranked_dists,
    rerank_scores,
    rerank_candidate_n,
    num_reranker_candidates,
    retrieval_relevance,
    retrieval_relevance_score,
    retrieval_explanation,
    prompt,
    retrieval_latency=None,
    dense_latency=None,
    bm25_latency=None,
    fusion_latency=None,
    rerank_latency=None,
    retrieval_eval_latency=None,
    groundedness_eval_latency=None,
    groundedness_label=None,
    groundedness_explanation=None,
    tokens=None,
    llm_latency=None
):
    lines = [
        "=== QUERY ===",
        f"Original: {original_query}",
        f"Rewritten: {rewritten_query}",
        f"Route: {retrieval_route}",
        f"Fusion alpha: {fusion_alpha:.2f}",
    ]

    lines.append("\n=== HYBRID RETRIEVAL (DENSE + BM25 FUSION) ===")
    for i, (doc, dist, fusion) in enumerate(raw_docs, start=1):
        cid = doc.metadata.get("chunk_id", "N/A")
        doc_name = doc.metadata.get("source_name", "N/A")
        page = doc.metadata.get("page", "N/A")
        sim = distance_to_similarity(dist)
        fusion_score = doc.metadata.get("fusion_score", fusion)

        lines.append(
            f"[Rank {i} | doc={doc_name} | chunk={cid} | page={page} | "
            f"dist={dist:.3f} | sim={sim:.3f} | fusion={fusion_score:.3f}]"
        )

    lines.append("\n=== RERANKED ORDER ===")
    for i, (doc, dist, score) in enumerate(
        zip(reranked_docs, reranked_dists, rerank_scores), start=1
    ):
        cid = doc.metadata.get("chunk_id", "N/A")
        doc_name = doc.metadata.get("source_name", "N/A")
        page = doc.metadata.get("page", "N/A")
        fusion_score = doc.metadata.get("fusion_score", 0)
        sim = distance_to_similarity(dist)

        lines.append(
            f"[Rank {i} | doc={doc_name} | chunk={cid} | page={page} | "
            f"dist={dist:.3f} | sim={sim:.3f} | fusion={fusion_score:.3f} | rerank={score:.3f}]"
        )

    lines.append("\n=== RETRIEVAL PIPELINE ===")
    lines.append(
        f"Fusion survivors: {len(raw_docs)}"
    )
    lines.append(
        f"Reranker candidate limit: {rerank_candidate_n}"
    )
    lines.append(
        f"Candidates sent to reranker: {num_reranker_candidates}"
    )
    lines.append(
        f"Final Top-K results: {len(reranked_docs)}"
    )

    lines.append("\n=== RETRIEVAL LATENCY ===")
    lines.append(
        f"Dense: {dense_latency:.3f}s"
        if dense_latency is not None
        else "Dense: N/A"
    )
    lines.append(
        f"BM25: {bm25_latency:.3f}s"
        if bm25_latency is not None
        else "BM25: N/A"
    )
    lines.append(
        f"Fusion: {fusion_latency:.3f}s"
        if fusion_latency is not None
        else "Fusion: N/A"
    )
    lines.append(
        f"Reranker: {rerank_latency:.3f}s"
        if rerank_latency is not None
        else "Reranker: N/A"
    )
    lines.append(
        f"Total retrieval: {retrieval_latency:.3f}s"
        if retrieval_latency is not None
        else "Total retrieval: N/A"
    )

    lines.append("\n=== RETRIEVAL RELEVANCE ===")
    lines.append(
        f"Relevance: {retrieval_relevance or 'N/A'}"
    )
    lines.append(
        f"Score: {retrieval_relevance_score:.3f}"
        if retrieval_relevance_score is not None
        else "Score: N/A"
    )
    lines.append(
        f"Explanation: {retrieval_explanation or 'N/A'}"
    )
    lines.append(
        f"Evaluation latency: {retrieval_eval_latency:.2f}s"
        if retrieval_eval_latency is not None
        else "Evaluation latency: N/A"
    )

    lines.append("\n=== GROUNDEDNESS CHECK ===")
    lines.append(f"Label: {groundedness_label or 'N/A'}")
    lines.append(f"Explanation: {groundedness_explanation or 'N/A'}")
    lines.append(f"Evaluation latency: {groundedness_eval_latency:.2f}s" if groundedness_eval_latency else "Evaluation latency: N/A")

    lines.append("\n=== FINAL PROMPT ===")
    lines.append(prompt)

    lines.append("\n=== LLM METRICS ===")
    if tokens:
        lines.append(f"Total tokens used: {tokens}")
    if llm_latency:
        lines.append(f"LLM llm_latency: {llm_latency:.2f}s")

    return "\n".join(lines)