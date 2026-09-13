def build_structured_chunk_text(doc):
    parts = []

    source_name = doc.metadata.get("source_name")
    section_header = doc.metadata.get("section_header")
    page = doc.metadata.get("page")
    doc_type = doc.metadata.get("doc_type")

    if source_name:
        parts.append(f"Source: {source_name}")
    if section_header:
        parts.append(f"Section: {section_header}")
    if page is not None:
        parts.append(f"Page: {page}")
    if doc_type and doc_type != "pdf":
        parts.append(f"Content Type: {doc_type}")

    parts.append(doc.page_content.strip())
    return "\n".join(parts).strip()