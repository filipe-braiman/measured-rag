"""Validated evaluation profiles, provenance, and portable artifact paths."""

import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath


EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
LEGACY_RESOLVED_CONFIG_FILENAME = "resolved_evaluation_config.json"
PROFILE_SCHEMA_VERSION = 1
PATH_BASE = "repository_root"
ALLOWED_RETRIEVAL_MODES = {"auto", "balanced", "semantic", "keyword"}
ALLOWED_REASONING_EFFORTS = {"none", "default", "low", "medium", "high"}
MODEL_DEFAULT_REASONING_EFFORT = {
    "qwen/qwen3.6-27b": "default",
    "qwen/qwen3.8-27b": "default",
    "openai/gpt-oss-120b": "medium",
}
JUDGE_PROFILE_FIELDS = {
    "judge_model",
    "judge_threshold",
    "judge_reasoning_effort",
    "judge_max_retries",
    "judge_max_completion_tokens",
    "judge_request_interval_seconds",
    "judge_retry_base_delay_seconds",
    "judge_retry_max_delay_seconds",
    "judge_retry_jitter_seconds",
    "judge_max_retry_after_seconds",
}
SECRET_PATTERNS = (
    (re.compile(r"\bgsk_[A-Za-z0-9_-]+\b", re.IGNORECASE), "<redacted>"),
    (re.compile(r"\bBearer\s+\S+", re.IGNORECASE), "Bearer <redacted>"),
    (
        re.compile(
            r"\b(api[_-]?key|authorization|token)(\s*[:=]\s*)(\S+)",
            re.IGNORECASE,
        ),
        r"\1\2<redacted>",
    ),
)


@dataclass(frozen=True)
class EvaluationConfig:
    profile_name: str
    profile_schema_version: int
    qa_data_path: str
    pdf_directory: str
    retrieval_mode: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_n: int
    fusion_threshold: float
    rerank_candidate_count: int
    rerank_top_k: int
    judge_model: str
    judge_threshold: float
    judge_reasoning_effort: str
    judge_max_retries: int
    judge_max_completion_tokens: int
    judge_request_interval_seconds: float
    judge_retry_base_delay_seconds: float
    judge_retry_max_delay_seconds: float
    judge_retry_jitter_seconds: float
    judge_max_retry_after_seconds: float
    generation_case_interval_seconds: float

    def as_dict(self):
        return asdict(self)

    @property
    def configuration_sha256(self):
        return sha256_json(self.as_dict())

    @property
    def qa_path(self):
        return resolve_repository_path(self.qa_data_path)

    @property
    def pdf_path(self):
        return resolve_repository_path(self.pdf_directory)

    @property
    def dataset_sha256(self):
        return sha256_file(self.qa_path)

    def with_judge_overrides(
        self,
        model=None,
        threshold=None,
        reasoning_effort=None,
        max_retries=None,
    ):
        updated = replace(
            self,
            judge_model=self.judge_model if model is None else model,
            judge_threshold=(
                self.judge_threshold if threshold is None else threshold
            ),
            judge_reasoning_effort=(
                self.judge_reasoning_effort
                if reasoning_effort is None
                else reasoning_effort
            ),
            judge_max_retries=(
                self.judge_max_retries
                if max_retries is None
                else max_retries
            ),
        )
        validate_profile(updated)
        return updated


def canonical_json(payload):
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(payload):
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def sha256_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {portable_hint(path)}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_repository_relative_path(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    if "\\" in value:
        raise ValueError(f"{field} must use POSIX path separators.")
    pure = PurePosixPath(value)
    if (
        not pure.parts
        or pure.is_absolute()
        or ".." in pure.parts
        or ":" in pure.parts[0]
    ):
        raise ValueError(f"{field} must be repository-relative.")
    return pure.as_posix()


def _require_string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")


def _require_positive_int(value, field):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")


def _require_threshold(value, field):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{field} must be between 0 and 1.")


def _require_finite_number(value, field, *, positive=False):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field} must be a finite {qualifier} number.")


def validate_profile(profile):
    if (
        not isinstance(profile.profile_schema_version, int)
        or isinstance(profile.profile_schema_version, bool)
        or profile.profile_schema_version != PROFILE_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported evaluation profile schema version: "
            f"{profile.profile_schema_version!r}."
        )
    _require_string(profile.profile_name, "profile_name")
    _validate_repository_relative_path(profile.qa_data_path, "qa_data_path")
    _validate_repository_relative_path(profile.pdf_directory, "pdf_directory")
    if profile.retrieval_mode not in ALLOWED_RETRIEVAL_MODES:
        raise ValueError(
            "retrieval_mode must be one of: "
            + ", ".join(sorted(ALLOWED_RETRIEVAL_MODES))
        )
    for field in (
        "chunk_size",
        "retrieval_top_n",
        "rerank_candidate_count",
        "rerank_top_k",
        "judge_max_completion_tokens",
    ):
        _require_positive_int(getattr(profile, field), field)
    if (
        not isinstance(profile.chunk_overlap, int)
        or isinstance(profile.chunk_overlap, bool)
        or profile.chunk_overlap < 0
    ):
        raise ValueError("chunk_overlap must be a nonnegative integer.")
    if profile.chunk_overlap >= profile.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")
    if profile.rerank_candidate_count > profile.retrieval_top_n:
        raise ValueError(
            "rerank_candidate_count cannot exceed retrieval_top_n."
        )
    if profile.rerank_top_k > profile.rerank_candidate_count:
        raise ValueError(
            "rerank_top_k cannot exceed rerank_candidate_count."
        )
    _require_threshold(profile.fusion_threshold, "fusion_threshold")
    _require_string(profile.judge_model, "judge_model")
    _require_threshold(profile.judge_threshold, "judge_threshold")
    if profile.judge_reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError(
            "judge_reasoning_effort must be one of: "
            + ", ".join(sorted(ALLOWED_REASONING_EFFORTS))
        )
    if (
        not isinstance(profile.judge_max_retries, int)
        or isinstance(profile.judge_max_retries, bool)
        or profile.judge_max_retries < 0
    ):
        raise ValueError("judge_max_retries must be a nonnegative integer.")
    _require_finite_number(
        profile.judge_request_interval_seconds,
        "judge_request_interval_seconds",
    )
    _require_finite_number(
        profile.judge_retry_base_delay_seconds,
        "judge_retry_base_delay_seconds",
        positive=True,
    )
    _require_finite_number(
        profile.judge_retry_max_delay_seconds,
        "judge_retry_max_delay_seconds",
        positive=True,
    )
    _require_finite_number(
        profile.judge_retry_jitter_seconds,
        "judge_retry_jitter_seconds",
    )
    _require_finite_number(
        profile.judge_max_retry_after_seconds,
        "judge_max_retry_after_seconds",
        positive=True,
    )
    _require_finite_number(
        profile.generation_case_interval_seconds,
        "generation_case_interval_seconds",
    )
    if (
        profile.judge_retry_base_delay_seconds
        > profile.judge_retry_max_delay_seconds
    ):
        raise ValueError(
            "judge_retry_base_delay_seconds cannot exceed "
            "judge_retry_max_delay_seconds."
        )


def default_reasoning_effort(model, fallback=None):
    return MODEL_DEFAULT_REASONING_EFFORT.get(model, fallback)


def generation_profile_configuration(profile):
    return {
        key: value
        for key, value in profile.as_dict().items()
        if key not in JUDGE_PROFILE_FIELDS
    }


def validate_generation_profile_scope(candidate, generation_profile):
    if generation_profile_configuration(candidate) != (
        generation_profile_configuration(generation_profile)
    ):
        raise ValueError(
            "Judge configuration does not match the frozen generation and "
            "benchmark profile."
        )
    if candidate.dataset_sha256 != generation_profile.dataset_sha256:
        raise ValueError(
            "Judge configuration dataset digest does not match the frozen "
            "generation dataset."
        )


def resolve_legacy_config_path(path=None, run_dir=None):
    if path is not None:
        candidate = Path(path)
        return (
            candidate.resolve()
            if candidate.is_absolute()
            else (REPO_ROOT / candidate).resolve()
        )
    if run_dir is not None:
        snapshot = Path(run_dir) / LEGACY_RESOLVED_CONFIG_FILENAME
        if snapshot.is_file():
            return snapshot.resolve()
    raise FileNotFoundError("No historical JSON evaluation configuration found.")


def evaluation_config_from_mapping(payload):
    if not isinstance(payload, dict):
        raise ValueError("Evaluation configuration must be an object.")
    expected = set(EvaluationConfig.__dataclass_fields__)
    unknown = payload.keys() - expected
    missing = expected - payload.keys()
    if unknown:
        raise ValueError(
            "Unknown evaluation configuration keys: "
            + ", ".join(sorted(unknown))
        )
    if missing:
        raise ValueError(
            "Missing evaluation configuration keys: "
            + ", ".join(sorted(missing))
        )
    config = EvaluationConfig(**payload)
    validate_profile(config)
    return config


def load_evaluation_config():
    from benchmarking.evaluation_settings import EVALUATION_CONFIG

    if not isinstance(EVALUATION_CONFIG, EvaluationConfig):
        raise TypeError(
            "benchmarking.evaluation_settings.EVALUATION_CONFIG must be an "
            "EvaluationConfig instance."
        )
    validate_profile(EVALUATION_CONFIG)
    return EVALUATION_CONFIG


def load_legacy_evaluation_config(path=None, run_dir=None):
    profile_path = resolve_legacy_config_path(path, run_dir=run_dir)
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"Evaluation profile not found: {portable_hint(profile_path)}"
        )
    try:
        with profile_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed evaluation profile: {portable_hint(profile_path)}"
        ) from exc
    return evaluation_config_from_mapping(payload)


def config_from_provenance(container):
    provenance = container.get("evaluation_profile")
    if not isinstance(provenance, dict):
        raise ValueError("Artifact has no embedded evaluation configuration.")
    return evaluation_config_from_mapping(
        provenance.get("resolved_configuration")
    )


def resolve_repository_path(value):
    portable = _validate_repository_relative_path(value, "artifact path")
    resolved = (REPO_ROOT / Path(*PurePosixPath(portable).parts)).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Configured path escapes the repository root.") from exc
    return resolved


def serialize_artifact_path(path):
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            "Persisted artifact paths must be inside the repository root."
        ) from exc
    return relative.as_posix()


def portable_hint(path):
    try:
        return serialize_artifact_path(path)
    except (OSError, ValueError):
        return "<external-path>"


def sanitize_error(error):
    message = " ".join(str(error).split()) or type(error).__name__
    repository_text = str(REPO_ROOT.resolve())
    message = message.replace(repository_text, "<repository_root>")
    message = message.replace(repository_text.replace("\\", "/"), "<repository_root>")
    message = re.sub(
        r"(?i)[A-Z]:\\Users\\[^\\\s]+(?:\\[^\s,;]+)*",
        "<redacted-path>",
        message,
    )
    message = re.sub(
        r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/][^\s,;]+",
        "<redacted-path>",
        message,
    )
    for pattern, replacement in SECRET_PATTERNS:
        message = pattern.sub(replacement, message)
    return message[:1000]


def portable_command(arguments):
    portable = []
    for index, argument in enumerate(map(str, arguments)):
        if index == 0:
            portable.append("python")
        elif Path(argument).is_absolute():
            portable.append(serialize_artifact_path(argument))
        else:
            portable.append(argument.replace("\\", "/"))
    return portable


def git_provenance():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    commit_hash = commit.stdout.strip() if commit.returncode == 0 else None
    dirty_state = bool(dirty.stdout) if dirty.returncode == 0 else None
    return commit_hash or None, dirty_state


def profile_provenance(profile, include_git=True):
    commit_hash, dirty_state = git_provenance() if include_git else (None, None)
    return {
        "profile_name": profile.profile_name,
        "profile_schema_version": profile.profile_schema_version,
        "resolved_configuration": profile.as_dict(),
        "configuration_sha256": profile.configuration_sha256,
        "dataset_sha256": profile.dataset_sha256,
        "git_commit": commit_hash,
        "git_dirty": dirty_state,
    }


def has_nonblank_records(path):
    path = Path(path)
    if not path.exists():
        return False
    if not path.is_file():
        raise ValueError(f"Expected a records file: {portable_hint(path)}")
    with path.open("r", encoding="utf-8") as handle:
        return any(line.strip() for line in handle)


def validate_profile_provenance(
    container,
    profile,
    records_path=None,
    allow_legacy_read=False,
):
    provenance = container.get("evaluation_profile")
    if provenance is None:
        if allow_legacy_read:
            return False
        if records_path is not None and has_nonblank_records(records_path):
            raise ValueError(
                "Cannot resume reproducibly: existing records lack evaluation "
                "profile and dataset provenance."
            )
        return False
    if not isinstance(provenance, dict):
        raise ValueError("evaluation_profile provenance must be an object.")
    expected = {
        "configuration_sha256": profile.configuration_sha256,
        "dataset_sha256": profile.dataset_sha256,
    }
    actual = {key: provenance.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            "Evaluation configuration or dataset digest does not match the "
            "recorded run provenance."
        )
    return True


def validate_generation_profile_provenance(
    container,
    profile,
    records_path=None,
    allow_legacy_read=False,
    allow_legacy_missing_case_pacing=False,
):
    """Validate only benchmark/generation scope plus the dataset digest."""
    provenance = container.get("evaluation_profile")
    if provenance is None:
        if allow_legacy_read:
            return False
        if records_path is not None and has_nonblank_records(records_path):
            raise ValueError(
                "Cannot reuse generation reproducibly: existing records lack "
                "evaluation profile and dataset provenance."
            )
        return False
    if not isinstance(provenance, dict):
        raise ValueError("evaluation_profile provenance must be an object.")
    recorded = provenance.get("resolved_configuration")
    if not isinstance(recorded, dict):
        raise ValueError("Artifact has no embedded evaluation configuration.")
    recorded_generation = {
        key: value
        for key, value in recorded.items()
        if key not in JUDGE_PROFILE_FIELDS
    }
    current_generation = generation_profile_configuration(profile)
    if (
        allow_legacy_missing_case_pacing
        and "generation_case_interval_seconds" not in recorded_generation
    ):
        current_generation = dict(current_generation)
        current_generation.pop("generation_case_interval_seconds", None)
        # The profile identity advanced with the pacing/schema revision; all
        # behavior-bearing legacy generation fields still have to match.
        current_generation["profile_name"] = recorded_generation.get(
            "profile_name"
        )
    if (
        recorded_generation != current_generation
        or provenance.get("dataset_sha256") != profile.dataset_sha256
    ):
        raise ValueError(
            "Generation/benchmark configuration or dataset digest does not "
            "match the recorded run provenance."
        )
    return True
