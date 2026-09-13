from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GENERATOR_MODEL = "openai/gpt-oss-120b"

REWRITER_MODEL = "openai/gpt-oss-20b"

GROUNDEDNESS_CHECKER = "openai/gpt-oss-20b"

RELEVANCE_CHECKER = "openai/gpt-oss-20b"

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2" # "BAAI/bge-reranker-base" is an alternative

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" # "sentence-transformers/all-MiniLM-L6-v2" is an alternative

# Gradio Initialization
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "local")  # "local" or "cloud"
SHARE = False

#DEMO DEPLOYMENT GUARDRAILS
CLOUD_MAX_DOCUMENTS = 3
CLOUD_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
CLOUD_MAX_TOTAL_UPLOAD_BYTES = 20 * 1024 * 1024
CLOUD_MAX_QUERY_CHARACTERS = 2000
CLOUD_MAX_QUERIES_PER_SESSION = 5
