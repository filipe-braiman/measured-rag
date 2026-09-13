import gradio as gr

from core.state import (
    new_kb_state,
    reset_kb_state,
)

from ingestion.indexing import (
    index_files,
    format_indexed_docs_status,
)

from utils.ids import new_conversation_id
from utils.file_utils import file_fingerprint
from deployment.guardrails import validate_upload


def upload_wrapper(
    files,
    previous_files,
    kb_state,
    mode_label,
    chunk_size,
    chunk_overlap,
):
    mode = "multi" if mode_label == "Multi-document" else "single"

    # Normalize current files
    current_files = []
    invalid_files = False

    if files:
        # Single-document mode may return one file/path,
        # while multi-document mode returns a list.
        if isinstance(files, str):
            files = [files]
        elif hasattr(files, "name"):
            files = [files]

        for f in files:
            if isinstance(f, str):
                current_files.append(f)
            elif hasattr(f, "name"):
                current_files.append(f.name)
            else:
                invalid_files = True

    invalid_files = invalid_files or any(not path for path in current_files)

    # Remove invalid/empty paths
    current_files = [
        path for path in current_files
        if path
    ]

    # Get fingerprints for current and previous uploader contents
    rejection = validate_upload(
        current_files, previous_files, kb_state, mode, chunk_size, chunk_overlap,
        invalid_files=invalid_files,
    )
    if rejection:
        # Only update status; preserve all session and uploader outputs.
        return (gr.skip(), gr.skip(), rejection, *(gr.skip() for _ in range(7)))

    current_fingerprints = set()

    for path in current_files:
        try:
            current_fingerprints.add(
                file_fingerprint(path)
            )
        except Exception:
            pass

    previous_fingerprints = set(previous_files or [])

    # CASE 1: Uploader is empty
    # Reset the entire KB.
    if not current_files:
        kb_state = reset_kb_state(
            kb_state or new_kb_state(mode=mode),
            mode=mode,
        )

        return (
            kb_state,                              # state_kb
            [],                                    # state_chat
            format_indexed_docs_status(kb_state), # file_status
            [],                                    # state_chunks
            gr.update(
                choices=[],
                value=[]
            ),                                     # selected_docs
            new_conversation_id(),                 # conversation
            "",                                    # sources
            "",                                    # debug
            [],                                    # state_uploaded_files
            gr.update(value=None),                 # file
        )

    # CASE 2: A previously uploaded file disappeared
    # Multi-document: Reset EVERYTHING.
    # Single-document:
    # The old file was replaced by another file.
    # Reset the old KB and index the new file.
    removed_files = previous_fingerprints - current_fingerprints

    if removed_files:

        # MULTI-DOCUMENT
        if mode == "multi":
            kb_state = reset_kb_state(
                kb_state or new_kb_state(mode=mode),
                mode=mode,
            )

            return (
                kb_state,
                [],
                format_indexed_docs_status(kb_state),
                [],
                gr.update(
                    choices=[],
                    value=[]
                ),
                new_conversation_id(),
                "",
                "",
                [],
                gr.update(value=None),
            )

        # SINGLE-DOCUMENT
        kb_state = reset_kb_state(
            kb_state or new_kb_state(mode=mode),
            mode=mode,
        )

        kb_state, status, chunks, doc_update = index_files(
            current_files,
            kb_state,
            mode,
            chunk_size,
            chunk_overlap,
        )

        current_state_files = []

        for path in current_files:
            try:
                current_state_files.append(
                    file_fingerprint(path)
                )
            except Exception:
                pass

        return (
            kb_state,
            gr.skip(),
            status,
            chunks,
            doc_update,
            gr.skip(),
            "",
            "",
            current_state_files,
            gr.skip(),
        )

    # CASE 3: No file was removed
    # In multi-document mode: append/index new files.
    # In single-document mode: index_files() already handles replacement.
    kb_state, status, chunks, doc_update = index_files(
        current_files,
        kb_state,
        mode,
        chunk_size,
        chunk_overlap,
    )

    return (
        kb_state,
        gr.skip(),              # keep chat
        status,
        chunks,
        doc_update,
        gr.skip(),              # keep conversation
        "",                     # clear sources
        "",                     # clear debug
        list(current_fingerprints),  # remember uploader contents
        gr.skip(),              # keep uploader visible
    )