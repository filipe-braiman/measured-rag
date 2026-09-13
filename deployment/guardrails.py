"""Cloud-only policy at upload and question callback boundaries."""

from functools import wraps
from pathlib import Path
import stat

from config import (
    DEPLOYMENT_MODE,
    CLOUD_MAX_DOCUMENTS,
    CLOUD_MAX_FILE_SIZE_BYTES,
    CLOUD_MAX_TOTAL_UPLOAD_BYTES,
    CLOUD_MAX_QUERY_CHARACTERS,
    CLOUD_MAX_QUERIES_PER_SESSION,
)
from utils.file_utils import file_fingerprint


def validate_upload(files, previous_files, kb_state, mode, chunk_size, chunk_overlap,
                    invalid_files=False):
    """Return a safe rejection message, or None; never mutate session state."""
    if DEPLOYMENT_MODE != "cloud":
        return None
    if invalid_files:
        return "An uploaded file is unavailable. Please upload it again."
    if not files:
        return None
    try:
        sizes = {}
        for path in files:
            if Path(path).suffix.lower() not in {".pdf", ".doc", ".docx"}:
                return "Please upload only PDF, DOC, or DOCX files."
            info = Path(path).stat()
            if not stat.S_ISREG(info.st_mode):
                return "An uploaded file is unavailable. Please upload it again."
            if info.st_size > CLOUD_MAX_FILE_SIZE_BYTES:
                return "Each file must be at most 10 MiB."
            # Check readability before the wrapper's fingerprint/reset logic.
            with open(path, "rb"):
                pass
            sizes[path] = info.st_size

        fingerprints = {path: file_fingerprint(path) for path in files}
        if mode == "multi" and set(previous_files or []) - set(fingerprints.values()):
            # Preserve the existing uploader-removal action: reset, no indexing.
            return None
        documents = (kb_state or {}).get("documents", {})
        if mode == "single" or (kb_state or {}).get("mode") != mode:
            documents = {}
        keys = set()
        total = 0
        for document in documents.values():
            # Existing metadata retains the server temporary path. No new
            # bookkeeping or document metadata is needed for accumulated size.
            info = Path(document["file_path"]).stat()
            if not stat.S_ISREG(info.st_mode):
                return "An indexed file is unavailable. Please reset and upload again."
            with open(document["file_path"], "rb"):
                pass
            total += info.st_size
            keys.add((document["fingerprint"], document["chunk_size"], document["chunk_overlap"]))
        count = len(documents)
        for path in files[:1] if mode == "single" else files:
            key = (fingerprints[path], chunk_size, chunk_overlap)
            if key in keys:
                continue
            keys.add(key)
            count += 1
            total += sizes[path]
        if count > CLOUD_MAX_DOCUMENTS:
            return "The cloud knowledge base supports at most 3 documents."
        if total > CLOUD_MAX_TOTAL_UPLOAD_BYTES:
            return "The cloud knowledge base supports at most 20 MiB of uploaded files."
    except (OSError, ValueError, TypeError, KeyError):
        return "An uploaded or indexed file is unavailable. Please upload it again."
    return None


def query_rejection(query):
    if DEPLOYMENT_MODE == "cloud" and len(query.strip()) > CLOUD_MAX_QUERY_CHARACTERS:
        return "Please shorten your question to 2,000 characters or fewer."
    return None


class QueryLimitError(ValueError):
    """The cloud session has no usable question allowance."""


def consume_query_allowance(count):
    """Return the next count without mutation; local sessions are unrestricted."""
    if DEPLOYMENT_MODE != "cloud":
        return count
    if type(count) is not int or not 0 <= count < CLOUD_MAX_QUERIES_PER_SESSION:
        raise QueryLimitError("This public demo allows up to 5 questions per session.")
    return count + 1


def guard_chat(chat):
    """Keep rejected pending questions out of the existing chat callback."""
    @wraps(chat)
    def guarded(*args, **kwargs):
        query = args[0] if args else kwargs.get("user_query", "")
        rejection = query_rejection(query) if query else None
        if DEPLOYMENT_MODE == "cloud" and (not query or not query.strip() or rejection):
            import gradio as gr
            if rejection:
                gr.Warning(rejection)
            return tuple(gr.skip() for _ in range(5))
        return chat(*args, **kwargs)
    return guarded
