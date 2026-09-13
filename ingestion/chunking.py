from langchain_core.documents import Document

from ingestion.loaders import load_document
from retrieval.formatting import build_structured_chunk_text

from ingestion.text_processing import (
    normalize_text,
    sliding_window_chunk,
)

from utils.file_utils import is_supported_file


def group_docs_by_section(docs):
    grouped = []
    current_group = []
    current_header = None

    for doc in docs:
        header = doc.metadata.get("section_header")

        # New header → start new group
        if header != current_header and current_group:
            grouped.append(current_group)
            current_group = []

        current_group.append(doc)
        current_header = header

    if current_group:
        grouped.append(current_group)

    return grouped

def prepare_chunks_for_file(file_path, chunk_size=700, chunk_overlap=70):
    if not is_supported_file(file_path):
        raise ValueError("Unsupported file type.")

    docs = load_document(file_path)

    if (
        docs
        and (
            docs[0].metadata.get("file_type", "").lower() == "pdf"
            or file_path.lower().endswith(".pdf")
        )
    ):
        chunks = chunk_pdf_documents(
            docs,
            chunk_size,
            chunk_overlap,
        )
    else:
        chunks = chunk_word_documents(
            docs,
            chunk_size,
            chunk_overlap,
        )

    return chunks

def chunk_pdf_documents(
    docs,
    chunk_size,
    chunk_overlap,
):
    chunks = []

    for doc in docs:
        raw_text = normalize_text(doc.page_content)
        if not raw_text:
            continue

        split_texts = sliding_window_chunk(
            raw_text,
            chunk_size,
            chunk_overlap,
        )

        for chunk_text in split_texts:
            new_doc = Document(
                page_content=chunk_text,
                metadata=doc.metadata.copy(),
            )
            new_doc.metadata["structured_context"] = (
                build_structured_chunk_text(new_doc)
            )
            chunks.append(new_doc)

    return chunks

def chunk_word_documents(
    docs,
    chunk_size,
    chunk_overlap,
):
    chunks = []

    grouped_docs = group_docs_by_section(docs)

    for group in grouped_docs:

        raw_text = normalize_text(
            "\n\n".join(
                d.page_content
                for d in group
                if d.page_content
            )
        )

        if not raw_text:
            continue

        base_metadata = group[0].metadata.copy()
        base_metadata["page"] = None

        split_texts = sliding_window_chunk(
            raw_text,
            chunk_size,
            chunk_overlap,
        )

        for chunk_text in split_texts:

            new_doc = Document(
                page_content=chunk_text,
                metadata=base_metadata.copy(),
            )

            new_doc.metadata["structured_context"] = (
                build_structured_chunk_text(new_doc)
            )

            chunks.append(new_doc)

    return chunks