import os
import time
import uuid

import gradio as gr

from langchain_community.vectorstores import Chroma

from core.models import get_embeddings

from rank_bm25 import BM25Okapi

from core.state import (
    new_kb_state,
    reset_kb_state,
)

from ingestion.metadata import build_document_metadata

from ingestion.chunking import prepare_chunks_for_file

from utils.file_utils import (
    file_fingerprint,
    is_supported_file,
)

def format_indexed_docs_status(kb_state):
    from html import escape
    import re
    docs = list(kb_state["documents"].values())
    if not docs:
        return "*No documents indexed.*"

    total_chunks = sum(d["chunk_count"] for d in docs)

    lines = [
        "**Knowledge base ready**\n",
        f"- **Mode:** {'Multi-document' if kb_state['mode'] == 'multi' else 'Single-document'}",
        f"- **Documents indexed:** {len(docs)}",
        f"- **Total chunks:** {total_chunks}",
    ]

    for i, d in enumerate(docs, start=1):
        # Treat filenames as literal text, including Markdown/HTML punctuation.
        file_name = re.sub(r"([\\`*_{}\[\]()#+.!|~-])", r"\\\1", escape(d["file_name"]))
        lines.append(
            f"- **Document {i}:** {file_name} | type={d['file_type'].upper()} | "
            f"chunks={d['chunk_count']} | size={d['chunk_size']} | overlap={d['chunk_overlap']}"
        )

    return "\n".join(lines)

def get_doc_name_to_id_map(kb_state):
    return {
        doc_meta["file_name"]: doc_id
        for doc_id, doc_meta in kb_state["documents"].items()
    }

def get_active_doc_ids(kb_state, scope_mode, selected_doc_names):
    all_doc_ids = set(kb_state["documents"].keys())

    if not all_doc_ids:
        return set()

    if scope_mode == "all":
        return all_doc_ids

    if scope_mode == "selected":
        if not selected_doc_names:
            return set()

        name_to_id = get_doc_name_to_id_map(kb_state)
        selected_ids = {name_to_id[name] for name in selected_doc_names if name in name_to_id}
        return selected_ids

    return all_doc_ids

def rebuild_bm25_index(kb_state):
    chunks = kb_state.get("bm25_chunks", [])
    if not chunks:
        kb_state["bm25_index"] = None
        return kb_state

    tokenized_chunks = [doc.page_content.split() for doc in chunks]
    kb_state["bm25_index"] = BM25Okapi(tokenized_chunks)
    return kb_state

def ensure_vectordb(kb_state):
    if kb_state["vectordb"] is not None:
        return kb_state

    collection_name = f"rag_session_{uuid.uuid4().hex[:8]}"
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
    )

    kb_state["vectordb"] = vectordb
    kb_state["collection_name"] = collection_name
    return kb_state

def index_files(files, kb_state, mode, chunk_size=700, chunk_overlap=100):
    """
    Single mode:
      - replaces current KB with first valid uploaded file only

    Multi mode:
      - appends all uploaded files
      - skips duplicates by fingerprint
    """
    if not files:
        return kb_state, "*No documents indexed.*", [], gr.update(choices=[], value=[])

    if mode not in {"single", "multi"}:
        mode = "single"

    # Normalize files into path list
    file_paths = []
    for f in files:
        if isinstance(f, str):
            file_paths.append(f)
        elif hasattr(f, "name"):
            file_paths.append(f.name)

    file_paths = [fp for fp in file_paths if fp]

    if not file_paths:
        return kb_state, "*No documents indexed.*", [], gr.update(choices=[], value=[])

    start = time.time()

    # Switching from old mode or enforcing single-mode replacement
    if kb_state is None or kb_state.get("mode") != mode:
        kb_state = reset_kb_state(kb_state or new_kb_state(), mode=mode)

    if mode == "single":
        # Replace entire KB with the first uploaded file
        kb_state = reset_kb_state(kb_state, mode="single")
        file_paths = [file_paths[0]]

    indexed_chunks_for_view = []
    indexed_count = 0
    skipped_duplicates = 0
    errors = []

    existing_fingerprints = {
        (d["fingerprint"], d["chunk_size"], d["chunk_overlap"])
        for d in kb_state["documents"].values()
    }

    kb_state = ensure_vectordb(kb_state)

    for file_path in file_paths:
        try:
            if not is_supported_file(file_path):
                errors.append(f"{os.path.basename(file_path)}: unsupported file type")
                continue

            fp = file_fingerprint(file_path)
            fp_key = (fp, chunk_size, chunk_overlap)

            if mode == "multi" and fp_key in existing_fingerprints:
                skipped_duplicates += 1
                continue

            chunks = prepare_chunks_for_file(
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            doc_meta, chunks = build_document_metadata(
                file_path=file_path,
                chunks=chunks,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            kb_state["vectordb"].add_documents(chunks)
            kb_state["documents"][doc_meta["doc_id"]] = doc_meta
            kb_state["bm25_chunks"].extend(chunks)

            indexed_chunks_for_view.extend(chunks)
            indexed_count += 1
            existing_fingerprints.add((doc_meta["fingerprint"], chunk_size, chunk_overlap))

        except Exception as e:
            errors.append(f"{os.path.basename(file_path)}: {str(e)}")

    kb_state = rebuild_bm25_index(kb_state)

    elapsed = time.time() - start
    status_text = format_indexed_docs_status(kb_state)
    status_text += f"\n\n- **Indexing time:** {elapsed:.2f} s"

    if mode == "single":
        status_text += "\n- **Upload behavior:** Replaces the current knowledge base"
    else:
        status_text += f"\n- **Newly indexed:** {indexed_count}"
        if skipped_duplicates:
            status_text += f"\n- **Skipped duplicates:** {skipped_duplicates}"

    if errors:
        from html import escape
        import re
        status_text += "\n\n**Errors**\n\n" + "\n".join(
            "- " + re.sub(r"([\\`*_{}\[\]()#+.!|~-])", r"\\\1", escape(error)).replace("\n", " ").replace("\r", " ")
            for error in errors[:5]
        )

    doc_choices = [meta["file_name"] for meta in kb_state["documents"].values()]
    return kb_state, status_text, kb_state["bm25_chunks"], gr.update(choices=doc_choices, value=[])