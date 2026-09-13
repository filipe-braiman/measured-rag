import os
import uuid
import json
from utils.helpers import truncate_text

from utils.file_utils import (
    file_fingerprint,
    get_file_extension,
)

def sanitize_metadata_value(value):
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(x, str) for x in value):
            return value
        if all(isinstance(x, int) for x in value):
            return value
        if all(isinstance(x, float) for x in value):
            return value
        if all(isinstance(x, bool) for x in value):
            return value
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    return str(value)

def sanitize_document_metadata(doc):
    allowed_keys = {
        "chunk_id",
        "doc_id",
        "page",
        "source",
        "source_name",
        "file_type",
        "doc_type",
        "section_header",
        "fusion_score",
        "raw_content",
        "category_depth",
        "languages",
        "filename",
        "fingerprint",
    }

    clean_metadata = {}
    for key, value in doc.metadata.items():
        if key not in allowed_keys:
            continue
        safe_value = sanitize_metadata_value(value)
        if safe_value is not None:
            clean_metadata[key] = safe_value

    doc.metadata = clean_metadata
    return doc

def enrich_chunk_metadata(
    chunks,
    doc_id,
    file_name,
    file_type,
    fingerprint,
):
    for i, chunk in enumerate(chunks):

        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["chunk_id"] = f"{doc_id}_{i}"
        chunk.metadata["file_type"] = (
            chunk.metadata.get("file_type", file_type)
        )
        chunk.metadata["source_name"] = (
            chunk.metadata.get("source_name", file_name)
        )
        chunk.metadata["filename"] = file_name
        chunk.metadata["fingerprint"] = fingerprint
        chunk.metadata["section_header"] = (
            chunk.metadata.get("section_header")
        )
        chunk.metadata["doc_type"] = (
            chunk.metadata.get("doc_type", file_type)
        )
        chunk.metadata["raw_content"] = truncate_text(
            chunk.page_content,
            1000,
        )

        sanitize_document_metadata(chunk)

    return chunks

def build_document_metadata(
    file_path,
    chunks,
    chunk_size,
    chunk_overlap,
):
    doc_id = uuid.uuid4().hex[:8]

    file_name = os.path.basename(file_path)
    file_type = get_file_extension(file_path).lstrip(".")
    fingerprint = file_fingerprint(file_path)

    chunks = enrich_chunk_metadata(
        chunks,
        doc_id,
        file_name,
        file_type,
        fingerprint,
    )

    doc_meta = {
        "doc_id": doc_id,
        "file_name": file_name,
        "file_path": file_path,
        "file_type": file_type,
        "fingerprint": fingerprint,
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }

    return doc_meta, chunks