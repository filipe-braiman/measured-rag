"""Network-denied image smoke test; never launch Gradio or use a provider client.

Run with Docker --network none and no credentials. The Python socket guard also
protects the Docker build check; network isolation at runtime covers subprocesses.
This verifies local infrastructure, not provider responses or hosted behavior.
"""

import argparse
import importlib
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import types


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    require(platform.system() == "Linux" and platform.machine() == "x86_64",
            "Run this verifier only inside the Linux amd64 image.")
    require(os.getuid() == 10001, "Expected the configured non-root runtime user.")
    require("GROQ_API_KEY" not in os.environ, "Do not supply credentials to verification.")
    require(os.environ.get("HF_HUB_OFFLINE") == "1"
            and os.environ.get("TRANSFORMERS_OFFLINE") == "1", "Offline mode is required.")
    root = Path(__file__).resolve().parents[1]
    require(root == Path("/app"), "Expected the image application directory.")
    for name in (".env", ".git", "benchmarking", "docs", "eval", "evaluation"):
        require(not (root / name).exists(), "Excluded content found in runtime image.")
    sys.path.insert(0, str(root))

    attempts = []

    def deny_network(*args, **kwargs):
        attempts.append("network")
        raise RuntimeError("Network access is prohibited in offline verification.")

    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.socket.sendto = deny_network
    socket.create_connection = deny_network
    socket.getaddrinfo = deny_network

    def deny_provider(*args, **kwargs):
        attempts.append("provider")
        raise RuntimeError("Provider access is prohibited in offline verification.")

    # The real core.llm constructs Groq at import. Leave its source untouched.
    stub = types.ModuleType("core.llm")
    stub.get_llm = deny_provider
    sys.modules["core.llm"] = stub

    # Check entry-point-only UI dependencies before heavyweight model checks.
    # Iterating copied modules alone cannot detect files omitted from COPY.
    from ui.controls import update_chunk_overlap_limit
    from ui.branding import LOGO_PATH, STYLESHEET_PATH
    require(callable(update_chunk_overlap_limit), "Missing UI control callback.")
    for asset in (LOGO_PATH, STYLESHEET_PATH):
        require(asset.is_relative_to(root) and asset.is_file(),
                "Missing UI runtime asset: " + asset.name)
        require(bool(asset.read_text(encoding="utf-8").strip()),
                "Empty UI runtime asset: " + asset.name)

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    pins = {}
    for line in (root / "constraints-app.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            req = Requirement(line)
            pins[canonicalize_name(req.name)] = req.specifier
    for dist in importlib.metadata.distributions():
        name = canonicalize_name(dist.metadata["Name"])
        require(name in pins and dist.version in pins[name],
                f"Unpinned or unexpected installed distribution: {name}")
    import torch
    require(torch.version.cuda is None and torch.__version__ == "2.10.0+cpu",
            "Expected CPU-only PyTorch.")

    from prefetch_models import MODEL_LICENSE_MARKERS, MODELS, check_application_models
    check_application_models(root / "config.py")
    cache = Path(os.environ["HF_HUB_CACHE"])
    for model_id, revision in MODELS.items():
        model_dir = cache / ("models--" + model_id.replace("/", "--"))
        require((model_dir / "refs" / "main").read_text().strip() == revision,
                "Unexpected offline model revision.")
        require((model_dir / "snapshots" / revision / "model.safetensors").is_file(),
                "Model weights are missing.")
        model_card = model_dir / "snapshots" / revision / "README.md"
        require(model_card.is_file()
                and MODEL_LICENSE_MARKERS[model_id] in model_card.read_text().lower(),
                "License-bearing model card is missing.")
    for notice in (root / "LICENSE", root / "THIRD_PARTY_NOTICES.md",
                   root / "third_party/licenses/Apache-2.0.txt",
                   root / "third_party/licenses/MIT-FlagEmbedding.txt"):
        require(notice.is_file() and notice.stat().st_size > 0,
                "Required license material is missing: " + notice.name)

    # Every copied runtime module except app.py and the provider constructor.
    modules = ["config", "groq", "gradio", "chromadb", "rank_bm25"]
    for directory in ("core", "chat", "generation", "retrieval", "ingestion",
                      "ui", "telemetry", "utils", "debug"):
        modules.extend(f"{directory}.{path.stem}" for path in
                       sorted((root / directory).glob("*.py"))
                       if path != root / "core" / "llm.py")
    for module in modules:
        importlib.import_module(module)
    require(os.environ.get("DEPLOYMENT_MODE") == "cloud",
            "Expected cloud telemetry mode in the image.")
    require(not (root / "answer_logs").exists(),
            "Cloud imports must not create a telemetry directory.")
    from core.models import initialize_models
    embeddings, reranker = initialize_models()
    vector = embeddings.embed_query("A synthetic document describes container verification.")
    require(len(vector) == 384 and all(math.isfinite(v) for v in vector),
            "Embedding inference failed.")
    scores = reranker.predict([("What is verified?", "The container is verified offline.")])
    require(len(scores) == 1 and math.isfinite(float(scores[0])), "Reranking failed.")

    import spacy
    require(len(spacy.load("en_core_web_sm")("This is a complete sentence.")) > 0,
            "Required spaCy asset is missing.")
    from docx import Document
    import fitz
    from ingestion.loaders import load_document
    sample = "The container verification checks that document text is available offline."
    with tempfile.TemporaryDirectory(prefix="rag-verify-") as tmp:
        folder = Path(tmp)
        pdf = folder / "sample.pdf"
        with fitz.open() as document:
            document.new_page().insert_text((72, 72), sample)
            document.save(pdf)
        docx = folder / "sample.docx"
        document = Document()
        document.add_paragraph(sample)
        document.save(docx)
        subprocess.run([
            "soffice", "-env:UserInstallation=" + (folder / "lo-profile").as_uri(),
            "--headless", "--convert-to", "doc:MS Word 97", "--outdir", tmp, str(docx),
        ], check=True, timeout=60, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for path in (pdf, docx, folder / "sample.doc"):
            require(path.is_file(), "Synthetic format conversion failed.")
            docs = load_document(str(path))
            require(any("container verification" in doc.page_content.lower() for doc in docs),
                    f"Offline {path.suffix} ingestion failed.")
    provider_attempt_count = attempts.count("provider")
    blocked_network_attempt_count = attempts.count("network")

    require(
        provider_attempt_count == 0,
        "Application verification attempted to construct or use the provider client.",
    )
    require(
        "app" not in sys.modules,
        "The interactive entry point must not be imported.",
    )

    print("PASS: runtime imports (provider stubbed); CPU embedding/reranker inference;")
    print("pinned offline models; spaCy; PDF/DOC/DOCX; non-root user; no provider calls.")
    print(
        "Blocked optional network attempts during verification:",
        blocked_network_attempt_count,
    )


if __name__ == "__main__":
    main()
