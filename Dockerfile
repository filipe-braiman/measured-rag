# Reproducible application image; validate the candidate before deployment.
# Cloud launch mode requires the platform-injected PORT at application launch.
# Build: docker build --platform linux/amd64 -t rag-foundation .
# Verify (does NOT launch app.py or use Groq):
# docker run --rm --network none --entrypoint python rag-foundation -B /app/deployment/verify_container.py
FROM python:3.11.14-slim-bookworm@sha256:83f339c1be6340ae1096010fdccf6552ac932d8f410d45d206014916bdf37e48

# This digest identifies the amd64 image. Fail rather than silently build ARM.
# A dated Debian archive pins the apt package universe as well as the base.
RUN test "$(dpkg --print-architecture)" = amd64 \
    && printf 'Types: deb\nURIs: https://snapshot.debian.org/archive/debian/20260901T000000Z/\nSuites: bookworm bookworm-updates\nComponents: main\nCheck-Valid-Until: no\n\nTypes: deb\nURIs: https://snapshot.debian.org/archive/debian-security/20260901T000000Z/\nSuites: bookworm-security\nComponents: main\nCheck-Valid-Until: no\n' > /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libmagic1 libgomp1 libreoffice-writer fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# LibreOffice is required by Unstructured's legacy DOC conversion path.
# libmagic: content detection; libgomp: native CPU inference; fonts: DOC layout.
# No OCR, GPU driver/runtime, notebook, Conda or extra application server.
ENV PYTHONDONTWRITEBYTECODE=1 \
    DEPLOYMENT_MODE=cloud \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/rag \
    TMPDIR=/runtime/tmp \
    GRADIO_TEMP_DIR=/runtime/gradio \
    XDG_CACHE_HOME=/runtime/cache \
    HF_HOME=/opt/models/huggingface \
    HF_HUB_CACHE=/opt/models/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/opt/models/huggingface/hub \
    HF_HUB_DISABLE_TELEMETRY=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app
RUN mkdir -p /runtime/tmp
COPY requirements-app.txt constraints-app.txt /app/
RUN python -m pip install pip==26.0.1 setuptools==80.10.2 wheel==0.46.3 \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu --constraint constraints-app.txt torch==2.10.0+cpu \
    && python -m pip install --no-build-isolation --requirement requirements-app.txt \
    && python -m pip check

COPY deployment/prefetch_models.py deployment/verify_container.py /app/deployment/
RUN mkdir -p /runtime/tmp /opt/models/huggingface/hub \
    && python -B /app/deployment/prefetch_models.py

# Copy redistribution notices after dependency and model acquisition so later
# build layers cannot obscure which notices accompany the shipped artifacts.
COPY LICENSE THIRD_PARTY_NOTICES.md /app/
COPY third_party/licenses/Apache-2.0.txt third_party/licenses/MIT-FlagEmbedding.txt /app/third_party/licenses/

# Offline cache lookup preserves the existing model IDs in config.py.
# Missing model assets must fail rather than downloading on a visitor request.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Explicit source allowlist. Never COPY . /app.
COPY app.py config.py /app/
COPY deployment/runtime.py /app/deployment/
COPY deployment/guardrails.py /app/deployment/
COPY core/llm.py core/models.py core/state.py /app/core/
COPY chat/service.py /app/chat/
COPY generation/answer_generator.py generation/groundedness.py generation/prompt.py generation/query_rewriter.py /app/generation/
COPY ingestion/chunking.py ingestion/indexing.py ingestion/loaders.py ingestion/metadata.py ingestion/text_processing.py /app/ingestion/
COPY retrieval/bm25.py retrieval/dense.py retrieval/formatting.py retrieval/fusion.py retrieval/relevance.py retrieval/reranking.py retrieval/routing.py /app/retrieval/
COPY telemetry/logger.py /app/telemetry/
COPY ui/branding.py ui/controls.py ui/reset.py ui/styles.css ui/upload.py ui/view_toggles.py /app/ui/
COPY assets/measured-rag-mark.svg assets/measured-rag-favicon-64.png /app/assets/
COPY assets/measured-rag-card.png assets/measured-rag-card-v3.png /app/assets/
COPY utils/file_utils.py utils/helpers.py utils/ids.py utils/source_formatter.py /app/utils/
COPY debug/builder.py /app/debug/

# Cloud telemetry writes operational JSON to stdout; only runtime caches and
# temporary files need writable directories.
RUN groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid rag --create-home --shell /usr/sbin/nologin rag \
    && mkdir -p /runtime/gradio /runtime/cache \
    && chown -R rag:rag /home/rag /runtime /opt/models
USER 10001:10001

# Offline verifier runs as the actual runtime user and blocks Python network access. It replaces core.llm with a fail-closed stub before importing services.
RUN python -B /app/deployment/verify_container.py
CMD ["python", "-B", "-u", "app.py"]
