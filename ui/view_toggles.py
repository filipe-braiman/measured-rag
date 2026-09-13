import gradio as gr


def render_indexed_chunks(chunks):
    """Refresh the always-visible inspection output using the existing format."""
    return toggle_chunks(True, chunks)


def toggle_chunks(show, chunks):
    if not show:
        return gr.update(visible=False)

    text = ""

    if chunks:
        text = "\n\n".join(
            (
                f"[Chunk {i+1} | doc {c.metadata.get('source_name', 'N/A')} | "
                f"chunk_id {c.metadata.get('chunk_id', 'N/A')} | "
                f"page {c.metadata.get('page', 'N/A')} | "
                f"section {c.metadata.get('section_header', 'N/A')} | "
                f"type {c.metadata.get('doc_type', 'N/A')}]\n"
                f"{c.page_content}"
            )
            for i, c in enumerate(chunks)
        )

    return gr.update(
        visible=True,
        value=text,
    )

def toggle_debug(show, debug_text):
    if not show:
        return gr.update(visible=False)

    return gr.update(
        visible=True,
        value=debug_text or "",
    )

def toggle_sources(show, sources):
    if not show:
        return gr.update(visible=False)

    return gr.update(
        visible=True,
        value=sources or "",
    )
