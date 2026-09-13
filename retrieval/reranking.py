from core.models import get_reranker
from retrieval.formatting import build_structured_chunk_text

def rerank_results(
    filtered_docs,
    rerank_model,
    rewritten_query,
    top_k,
):
    if not filtered_docs:
        return [], [], []

    top_k = max(1, int(top_k))

    if rerank_model is None:
        rerank_model = get_reranker()

    # CrossEncoder evaluates query-passage pairs jointly.
    pairs = [
        (
            rewritten_query,
            build_structured_chunk_text(doc),
        )
        for doc, _, _ in filtered_docs
    ]

    scores = rerank_model.predict(
        pairs,
        show_progress_bar=False,
    )

    rerank_scores = [
        float(score)
        for score in scores
    ]

    rerank_tuples = [
        (
            doc,
            dist,
            score,
            fusion,
        )
        for (doc, dist, fusion), score
        in zip(filtered_docs, rerank_scores)
    ]

    # Highest relevance first
    rerank_tuples.sort(
        key=lambda x: x[2],
        reverse=True,
    )

    # Keep final Top-K
    rerank_tuples = rerank_tuples[:top_k]

    reranked_docs = [
        doc
        for doc, _, _, _ in rerank_tuples
    ]

    reranked_dists = [
        dist
        for _, dist, _, _ in rerank_tuples
    ]

    reranked_scores = [
        score
        for _, _, score, _ in rerank_tuples
    ]

    return (
        reranked_docs,
        reranked_dists,
        reranked_scores,
    )