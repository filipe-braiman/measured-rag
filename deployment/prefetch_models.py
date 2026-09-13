"""Build-only downloads of the two frozen local model snapshots.
Never import config.py/app.py here: dotenv, telemetry and provider construction are not part of an image build. The revisions below were observed in the local working model cache during the audit, not inferred from a moving Hub branch.
"""

import argparse
import ast
import os
from pathlib import Path


MODELS = {
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "cross-encoder/ms-marco-MiniLM-L6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
}

MODEL_LICENSE_MARKERS = {
    "BAAI/bge-small-en-v1.5": "license: mit",
    "cross-encoder/ms-marco-MiniLM-L6-v2": "license: apache-2.0",
}


def check_application_models(config_path):
    """Read constants without loading dotenv or importing application modules."""
    values = {}
    for node in ast.parse(config_path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value.value
    actual = {values["EMBEDDING_MODEL_NAME"], values["RERANK_MODEL_NAME"]}
    if actual != set(MODELS):
        raise RuntimeError("Application model IDs differ from the build manifest.")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    from huggingface_hub import snapshot_download

    cache = Path(os.environ["HF_HUB_CACHE"])
    for model_id, revision in MODELS.items():
        snapshot = Path(snapshot_download(
            model_id,
            revision=revision,
            cache_dir=cache,
            token=False,
            # BERT tokenizer, sentence-transformer pooling and safetensors.
            # Do not pull alternate ONNX/OpenVINO weights or repository code.
            allow_patterns=["*.json", "*.txt", "*.safetensors", "README.md"],
        ))
        if snapshot.name != revision or not (snapshot / "model.safetensors").is_file():
            raise RuntimeError("Incomplete or unexpected model snapshot.")
        model_card = snapshot / "README.md"
        if (
            not model_card.is_file()
            or MODEL_LICENSE_MARKERS[model_id]
            not in model_card.read_text(encoding="utf-8").lower()
        ):
            raise RuntimeError(f"Missing license-bearing model card for {model_id}.")
        # Existing constructors request model IDs without a revision. In an offline image, map their default ref to this immutable snapshot.
        refs = snapshot.parent.parent / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text(revision, encoding="utf-8")
        print(f"Provisioned {model_id} at {revision}")


if __name__ == "__main__":
    main()
