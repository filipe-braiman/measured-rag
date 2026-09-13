from utils.helpers import distance_to_similarity


def hybrid_fusion(
    dense_results,
    bm25_scores,
    chunk_lookup,
    fusion_alpha,
    score_threshold,
):
    dense_raw = {}

    # 1. Collect raw dense similarities
    for doc, dist in dense_results:
        cid = doc.metadata.get("chunk_id")

        dense_raw[cid] = (
            doc,
            distance_to_similarity(dist),
            dist,
        )

    # 2. Min-max normalize dense similarities to [0, 1]
    dense_scores = {}

    if dense_raw:
        similarities = [
            sim
            for _, sim, _
            in dense_raw.values()
        ]

        min_sim = min(similarities)
        max_sim = max(similarities)

        for cid, (doc, sim, dist) in dense_raw.items():

            if max_sim == min_sim:
                normalized_sim = 1.0
            else:
                normalized_sim = (
                    (sim - min_sim)
                    / (max_sim - min_sim)
                )

            dense_scores[cid] = (
                doc,
                normalized_sim,
                dist,
            )

    # 3. Fuse normalized dense + normalized BM25 scores
    fusion_results = {}

    all_chunk_ids = (
        set(dense_scores.keys())
        | set(bm25_scores.keys())
    )

    for cid in all_chunk_ids:

        dense_sim = 0.0
        dist = 1.0

        if cid in dense_scores:
            doc, dense_sim, dist = dense_scores[cid]
        else:
            doc = chunk_lookup.get(cid)

            if doc is None:
                continue

        bm25_sim = bm25_scores.get(cid, 0.0)

        final_score = (
            fusion_alpha * dense_sim
            + (1 - fusion_alpha) * bm25_sim
        )

        doc.metadata["fusion_score"] = final_score

        fusion_results[cid] = (
            doc,
            final_score,
            dist,
        )

    # 4. Rank by fusion score
    ranked = sorted(
        fusion_results.values(),
        key=lambda x: x[1],
        reverse=True,
    )

    # 5. Apply candidate threshold
    combined_docs = [
        (
            doc,
            dist,
            fusion_score,
        )
        for doc, fusion_score, dist in ranked
        if (
            score_threshold <= 0
            or fusion_score >= score_threshold
        )
    ]

    return combined_docs