import gradio as gr
from ingestion.indexing import format_indexed_docs_status
from utils.ids import new_conversation_id
from core.state import (
    new_kb_state,
    reset_kb_state,
)

def reset_all(kb_state, mode):
    kb_state = reset_kb_state(kb_state or new_kb_state(mode=mode), mode=mode)

    return (
        kb_state,                             # state_kb
        [],                                   # state_chat
        [],                                   # state_chunks
        format_indexed_docs_status(kb_state), # file_status
        "",                                   # sources
        "",                                   # debug_box
        new_conversation_id(),                # state_conversation_id
        gr.update(value=None),                # file
        gr.update(choices=[], value=[], visible=False),  # selected_docs
        gr.update(value="all", visible=(mode == "multi")),  # query_scope_mode
    )

def reset_chat_only(kb_state):
    return (
        kb_state,                      # KEEP KB
        [],                            # clear chat
        kb_state["bm25_chunks"],       # keep chunks
        format_indexed_docs_status(kb_state),
        "",                            # sources
        "",                            # debug
        new_conversation_id(),
    )

def on_mode_change(mode_label, kb_state):
    mode = "multi" if mode_label == "Multi-document" else "single"
    kb_state = reset_kb_state(kb_state or new_kb_state(mode=mode), mode=mode)
    upload_label = "Upload Documents" if mode == "multi" else "Upload Document"
    upload_file_count = "multiple" if mode == "multi" else "single"

    return (
        kb_state,                             # state_kb
        [],                                   # state_chat
        [],                                   # state_chunks
        format_indexed_docs_status(kb_state), # file_status
        "",                                   # sources
        "",                                   # debug_box
        new_conversation_id(),                # state_conversation_id
        gr.update(
            value=None,
            label=upload_label,
            file_count=upload_file_count,
        ),        # file
        gr.update(choices=[], value=[], visible=False),   # selected_docs
        gr.update(value="all", visible=(mode == "multi")),# query_scope_mode
    )

def reset_wrapper(kb_state, mode_label):
    mode = "multi" if mode_label == "Multi-document" else "single"
    return reset_all(kb_state, mode)

def on_scope_change(scope_mode):
    return gr.update(visible=(scope_mode == "selected"))