"""Read-only static checks for the application container build inputs.

This validates our restricted Dockerfile/ignore grammar, dependency pins, source syntax and local import coverage. It is not a Docker parser or a Linux resolver; an actual build and network-isolated container verification remain required.
No application imports, credentials, downloads or generated artifacts are used.
"""

import argparse
import ast
from pathlib import Path
import re
import shlex
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from prefetch_models import check_application_models

    root = Path(__file__).resolve().parents[1]
    pins = {}
    for line in (root / "constraints-app.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        req = Requirement(line)
        name = canonicalize_name(req.name)
        require(name not in pins and re.fullmatch(r"==[^,*]+", str(req.specifier)),
                "Constraints must contain unique exact version pins.")
        require(not req.url and not req.marker and not req.extras,
                "Unexpected constraint syntax.")
        require(not name.startswith(("nvidia-", "cuda-", "jupyter", "notebook"))
                and name not in {"triton", "deepeval", "datasets", "ipykernel"},
                "Non-application dependency found.")
        pins[name] = req.specifier
    direct = set()
    for line in (root / "requirements-app.txt").read_text().splitlines():
        if not line or line.startswith("#") or line == "-c constraints-app.txt":
            continue
        req = Requirement(line)
        name = canonicalize_name(req.name)
        require(name not in direct and name in pins, "Missing or duplicate dependency pin.")
        direct.add(name)
        if req.url:
            require(name == "en-core-web-sm" and re.search(r"#sha256=[a-f0-9]{64}$", req.url),
                    "Expected hashed spaCy model wheel.")
        else:
            require(req.specifier == pins[name], "Direct dependency differs from constraint.")

    docker = (root / "Dockerfile").read_text()
    require(re.search(r"^FROM python:3\.11\.14-slim-bookworm@sha256:[a-f0-9]{64}$",
                      docker, re.M), "Base must be pinned by digest.")
    require('CMD ["python", "-B", "-u", "app.py"]' in docker
            and docker.count("\nCMD ") == 1 and "\nUSER 10001:10001\n" in docker,
            "Expected one exec-form application process and non-root user.")
    require("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in docker
            and "https://download.pytorch.org/whl/cpu" in docker,
            "Offline and CPU installation controls missing.")
    sources = set()
    for line in docker.replace("\\\n", " ").splitlines():
        if not line or line.startswith("#"):
            continue
        instruction = line.split()[0]
        require(instruction in {"FROM", "RUN", "ENV", "WORKDIR", "COPY", "USER", "CMD"},
                "Unexpected Dockerfile instruction or broken continuation.")
        if instruction == "COPY":
            parts = shlex.split(line)
            require(parts[-1].startswith("/app/"), "Unexpected COPY destination.")
            for source in parts[1:-1]:
                require(not any(c in source for c in "*?[]")
                        and (root / source).is_file(), "COPY must list existing files explicitly.")
                sources.add(source)
                require(parts[-1] == "/app/" + source.rpartition("/")[0] + "/"
                        if "/" in source else parts[-1] == "/app/",
                        "COPY must preserve repository runtime paths: " + source)

    for asset in ("ui/styles.css", "assets/measured-rag-mark.svg"):
        require(asset in sources, "Missing UI runtime asset: " + asset)

    ignore = (root / ".dockerignore").read_text()
    require(ignore == (root / ".gcloudignore").read_text(), "Build context policies differ.")
    rules = [line for line in ignore.splitlines() if line and not line.startswith("#")]
    require(rules[0] == "**", "Build contexts must deny files by default.")
    allowed = set()
    directories = set()
    for index, rule in enumerate(rules[1:], start=1):
        require(rule.startswith("!") and not any(c in rule for c in "*?[]"),
                "Only exact allowlist exceptions are permitted.")
        path = rule[1:]
        require(not Path(path).is_absolute() and ".." not in Path(path).parts,
                "Allowlist paths must stay in the repository.")
        if path.endswith("/"):
            # Cloud Build's directory traversal uses these exact exceptions.
            # Keep the root deny rule and explicit file/COPY allowlists; do not reintroduce directory/** rules that excluded required build files.
            directories.add(path[:-1])
        else:
            allowed.add(path)
    expected = sources | {"Dockerfile", ".dockerignore", ".gcloudignore",
                          "deployment/verify_foundation.py"}
    require(allowed == expected, "Context must contain exactly the explicit build inputs.")
    for path in allowed:
        require((root / path).is_file(), "Missing allowlisted build input.")
        require("/" not in path or str(Path(path).parent).replace("\\", "/") in directories,
                "An allowlisted file has an excluded parent directory.")
    for forbidden in (".env", ".env.production", ".git/config", "core/.env", "core/secret.py",
                      "docs/configuration.md", "docs/user-guide.md",
                      "benchmarking/data/paper.pdf", "benchmarking/runs/run/answers.jsonl",
                      "answer_logs/private.jsonl", "models/model.safetensors", "ui/__pycache__/x.pyc"):
        require(forbidden not in allowed, "Private or generated file admitted to context.")

    # Check local runtime import closure without importing app.py or config.py.
    packages = {path.split("/")[0] for path in sources if "/" in path}
    packages |= {"config", "app"}
    count = 0
    for path in sorted(sources | {"deployment/verify_foundation.py"}):
        if not path.endswith(".py"):
            continue
        tree = ast.parse((root / path).read_text(encoding="utf-8-sig"), filename=path)
        count += 1
        if path.startswith("deployment/"):
            continue
        for node in ast.walk(tree):
            imports = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                       else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for module in imports:
                if module.split(".")[0] in packages:
                    candidate = module.replace(".", "/") + ".py"
                    require(candidate in sources, "Missing local runtime import: " + module)
    check_application_models(root / "config.py")
    print(f"PASS: {len(pins)} dependency pins; {len(allowed)} exact build inputs; {count} Python sources.")
    print("PASS: restricted Dockerfile structure, context exclusions, runtime import closure, model IDs.")
    print("Not verified here: Docker parsing/build, Linux dependency resolution, image runtime or size.")


if __name__ == "__main__":
    main()
