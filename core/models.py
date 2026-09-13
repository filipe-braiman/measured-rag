from config import (
    EMBEDDING_MODEL_NAME,
    GENERATOR_MODEL,
    GROUNDEDNESS_CHECKER,
    RELEVANCE_CHECKER,
    RERANK_MODEL_NAME,
    REWRITER_MODEL,
)


GENERATOR_REQUEST_SETTINGS = {
    "temperature": 0.5,
    "max_completion_tokens": 1536,
    "reasoning_effort": "low",
    "include_reasoning": False,
}


REWRITER_REQUEST_SETTINGS = {
    "temperature": 0.0,
    "max_tokens": 200,
}


RELEVANCE_CHECKER_REQUEST_SETTINGS = {
    "temperature": 0.0,
    "seed": None, #7331 for benchmarking
    "max_completion_tokens": 512,
    "reasoning_effort": "low",
    "reasoning_format": "hidden",
    "strict_schema_mode": "json_schema_strict",
    "max_retries": 2,
    "timeout_seconds": 45.0,
}


GROUNDEDNESS_CHECKER_REQUEST_SETTINGS = {
    "temperature": 0.0,
    "seed": None, #7331 for benchmarking
    "max_completion_tokens": 512,
    "reasoning_effort": "low",
    "reasoning_format": "hidden",
    "strict_schema_mode": "json_schema_strict",
    "max_retries": 2,
    "timeout_seconds": 45.0,
}

SUPPORTED_CHECKER_MODELS = {
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
}


def generation_request_settings():
    return {
        "model": GENERATOR_MODEL,
        **GENERATOR_REQUEST_SETTINGS,
    }


def rewriter_request_settings():
    return {
        "model": REWRITER_MODEL,
        **REWRITER_REQUEST_SETTINGS,
    }


def relevance_checker_request_settings(settings_override=None):
    if settings_override is not None:
        return dict(settings_override)

    return {
        "model": RELEVANCE_CHECKER,
        **RELEVANCE_CHECKER_REQUEST_SETTINGS,
    }


def groundedness_checker_request_settings(settings_override=None):
    if settings_override is not None:
        return dict(settings_override)

    return {
        "model": GROUNDEDNESS_CHECKER,
        **GROUNDEDNESS_CHECKER_REQUEST_SETTINGS,
    }

_embeddings = None
_rerank_model = None


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        print(
            f"[Models] Loading embedding model: "
            f"{EMBEDDING_MODEL_NAME}"
        )

        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

        print("[Models] Embedding model loaded.")

    return _embeddings

def get_reranker():
    global _rerank_model

    if _rerank_model is None:
        from sentence_transformers import CrossEncoder

        print(
            f"[Models] Loading reranker model: "
            f"{RERANK_MODEL_NAME}"
        )

        _rerank_model = CrossEncoder(
            RERANK_MODEL_NAME,
            #backend="openvino", # for performance, can be changed if needed
        )

        print("[Models] Reranker model loaded.")

    return _rerank_model

def initialize_models():
    """
    Load all local ML models during application startup.

    Models remain cached in the module-level singleton variables
    and are reused throughout the application lifecycle.
    """
    print("[Models] Initializing application models...")

    embeddings = get_embeddings()
    reranker = get_reranker()

    print("[Models] All models initialized.")

    return embeddings, reranker
