def new_kb_state(mode="single"):
    return {
        "mode": mode,                     # "single" or "multi"
        "vectordb": None,
        "collection_name": None,
        "documents": {},                  # doc_id -> metadata dict
        "bm25_index": None,
        "bm25_chunks": [],
        "rerank_model": None,
    }

def delete_vectorstore(vectordb):
    if not vectordb:
        return
    try:
        vectordb._client.delete_collection(name=vectordb._collection.name)
    except Exception:
        pass

def reset_kb_state(kb_state, mode="single"):
    if kb_state and kb_state.get("vectordb"):
        delete_vectorstore(kb_state["vectordb"])
    return new_kb_state(mode=mode)