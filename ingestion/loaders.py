import os

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
)

from utils.file_utils import get_file_extension

def load_pdf_document(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    file_name = os.path.basename(file_path)

    for doc in docs:
        doc.metadata["source_name"] = file_name
        doc.metadata["file_type"] = "pdf"
        doc.metadata["doc_type"] = "pdf"
        doc.metadata["section_header"] = None

        if "page" in doc.metadata and doc.metadata["page"] is not None:
            doc.metadata["page"] += 1

    return docs

def load_word_document(file_path):
    loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
    elements = loader.load()

    file_name = os.path.basename(file_path)
    file_type = get_file_extension(file_path).lstrip(".")

    docs = []
    current_header = None
    current_section_parts = []

    def flush_section():
        nonlocal current_section_parts, current_header, docs
        if not current_section_parts:
            return

        section_text = "\n".join(current_section_parts).strip()
        if not section_text:
            current_section_parts = []
            return

        docs.append(
            Document(
                page_content=section_text,
                metadata={
                    "source_name": file_name,
                    "file_type": file_type,
                    "doc_type": "section",
                    "section_header": current_header,
                    "page": None,
                },
            )
        )
        current_section_parts = []

    for el in elements:
        text = (el.page_content or "").strip()
        if not text:
            continue

        element_type = el.metadata.get("category") or el.metadata.get("type") or "Unknown"

        if element_type in {"Title", "Header"}:
            flush_section()
            current_header = text
        else:
            current_section_parts.append(text)

    flush_section()

    return docs

def load_document(file_path):
    ext = get_file_extension(file_path)

    if ext == ".pdf":
        return load_pdf_document(file_path)
    elif ext in {".doc", ".docx"}:
        return load_word_document(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
