# Development

This guide owns the local application-development setup for Measured RAG. Run
commands from the repository root, which contains `app.py` and `config.py`.
Benchmark scoring and judging use a separate environment documented in
[Benchmarking](benchmarking.md#evaluation-environment).

## Prerequisites

- CPython 3.11. The deployment image pins Python 3.11.14; local setup supports
  the Python 3.11 series.
- Git.
- A Groq API key for language-model and runtime diagnostic requests.
- Internet access during installation and the first local model load.
- LibreOffice for legacy `.doc` ingestion and the native `libmagic` library
  used by Unstructured. Install these system dependencies through the operating
  system when needed; the Linux container includes them separately.

## Create the application environment

### Windows PowerShell with `venv`

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, review the current execution policy before
using a process-scoped policy change. Activation is optional; commands can use
`.\.venv\Scripts\python.exe` directly.

### Linux and macOS with `venv`

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Conda alternative

Pip remains the installer for the repository's authoritative dependency file:

```bash
conda create -n measured-rag-app python=3.11
conda activate measured-rag-app
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` supports the local UI and the production ingestion,
retrieval, and generation path. It also supports
`python -m benchmarking.generate_answers`. DeepEval and dataset-construction
dependencies deliberately belong to the isolated evaluation environment.

The Docker/Cloud Run installation is a separate Linux/amd64 deployment
contract owned by `requirements-app.txt`, `constraints-app.txt`, and the
`Dockerfile`. Do not use that container-specific installation as the normal
desktop setup. See [Deployment](deployment.md) for the public deployment
architecture and hosted operational posture.

## Configure environment variables

Copy the safe template before adding a provider credential.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Linux and macOS:

```bash
cp .env.example .env
```

Set `GROQ_API_KEY` in the local `.env` and keep `DEPLOYMENT_MODE=local`. Never
commit `.env`; it is ignored by Git. Local mode still sends answer generation,
history-aware query rewriting, retrieval relevance, and groundedness requests to
Groq. See [Configuration](configuration.md#application-read-environment-variables)
for the canonical variable semantics.

## Run the application

```bash
python app.py
```

Startup initializes the local embedding and cross-encoder reranking models.
Their assets download from Hugging Face on first use and are cached locally, so
the first start can take longer and use additional disk space. Local model
execution does not make the complete application offline because Groq still
serves the language-model and diagnostic requests. Successful local answer paths
also append content-bearing records under `answer_logs/`; review the
[telemetry privacy and handling guidance](telemetry.md#privacy-and-safe-handling)
before inspecting or sharing them.

## Run offline tests

The existing tests use Python's built-in `unittest` and local fixtures. The
explicit module command is required because repository-root discovery does not
discover tests in the current namespace-style directories.

```bash
python -m unittest deployment.test_guardrails deployment.test_runtime deployment.test_session_state telemetry.test_logger ui.test_inspection ui.test_selection
```

These tests do not launch `app.py`, download models, or make provider calls.
Target one owning directory when iterating on a narrower change:

```bash
python -m unittest discover -s deployment -p "test_*.py"
python -m unittest discover -s telemetry -p "test_*.py"
python -m unittest discover -s ui -p "test_*.py"
```

## Project structure and change boundaries

| Path | Responsibility |
| --- | --- |
| `app.py`, `ui/` | Gradio construction, event wiring, and presentation |
| `ingestion/`, `retrieval/`, `generation/`, `chat/` | Production document RAG path |
| `core/`, `telemetry/`, `deployment/` | Shared state and models, operational records, and hosted policy |
| `benchmarking/` | Evaluation stages that exchange repository-contained artifacts |
| `docs/` | Canonical project documentation and publication guidance |

Read [Architecture](architecture.md) before changing component boundaries.
Runtime controls belong in [Configuration](configuration.md), user-visible
behavior in the [User guide](user-guide.md), and evaluation methodology in
[Benchmarking](benchmarking.md). Keep code, generated benchmark artifacts,
deployment changes, and documentation changes independently reviewable. Never
import `app.py` from headless tests or benchmark modules.

## Platform notes

- Windows activates `.\.venv\Scripts\Activate.ps1`; Linux and macOS use
  `source .venv/bin/activate`.
- Run commands from the repository root. Quote paths passed to external tools
  when they contain spaces.
- OneDrive and similar synchronizers can temporarily lock vector stores or
  benchmark files. Close processes holding a file and retry the validated
  operation; do not hand-edit benchmark JSON or JSONL as a repair.
