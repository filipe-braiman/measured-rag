from utils.helpers import distance_to_similarity

def format_sources(
    docs,
    dists,
    rerank_scores,
):
    sources = []
    for i, (doc, dist, score) in enumerate(zip(docs, dists, rerank_scores), start=1):
        sim = distance_to_similarity(dist)
        cid = doc.metadata.get("chunk_id", "N/A")
        page = doc.metadata.get("page", "N/A")
        fusion = doc.metadata.get("fusion_score", 0)
        section_header = doc.metadata.get("section_header", "N/A")
        doc_type = doc.metadata.get("doc_type", "N/A")
        file_type = doc.metadata.get("file_type", "N/A")
        doc_name = doc.metadata.get("source_name", "N/A")

        sources.append(
            f"[Source {i} | doc={doc_name} | chunk={cid} | page={page} | file_type={file_type} | "
            f"type={doc_type} | section={section_header} | dist={dist:.3f} | "
            f"sim={sim:.3f} | fusion={fusion:.3f} | rerank={score:.3f}]\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n".join(sources)