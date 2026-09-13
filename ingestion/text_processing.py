def normalize_text(text):
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()

def sliding_window_chunk(text, size, overlap):
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("Chunk size must be a positive integer.")

    if (
        isinstance(overlap, bool)
        or not isinstance(overlap, int)
        or overlap < 0
    ):
        raise ValueError("Chunk overlap must be a nonnegative integer.")

    if overlap >= size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + size

        if end >= text_len:
            final_chunk = text[start:].strip()
            if final_chunk:
                chunks.append(final_chunk)
            break

        safe_end = end

        while safe_end > start and text[safe_end] not in (" ", "\n"):
            safe_end -= 1

        if safe_end == start:
            safe_end = end
            while (
                safe_end < text_len
                and text[safe_end] not in (" ", "\n")
            ):
                safe_end += 1

            if safe_end >= text_len:
                safe_end = text_len

        chunk_text = text[start:safe_end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        next_start = safe_end - overlap

        # Word-boundary adjustment can occasionally make the configured overlap larger than the actual chunk. Fall back to no overlap for this boundary rather than moving backward or looping indefinitely.
        if next_start <= start:
            next_start = safe_end

        if next_start <= start:
            raise RuntimeError("Chunking failed to advance through the text.")

        start = next_start

    return chunks