def update_chunk_overlap_limit(chunk_size, chunk_overlap):
    chunk_size = int(chunk_size)
    chunk_overlap = int(chunk_overlap)
    CHUNK_OVERLAP_STEP = 10
    import gradio as gr

    maximum = max(0, chunk_size // 2)
    value = min(chunk_overlap, maximum)

    return gr.Slider(maximum=maximum, value=value)