"""Deterministic Markdown reporting over existing QASPER run artifacts."""

import argparse
import hashlib
import html
import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from benchmarking.artifact_io import atomic_write_text
from benchmarking.checker_diagnostics import aggregate_checker_diagnostics
from benchmarking.evaluation_config import REPO_ROOT, canonical_json, sha256_file, sha256_json
from benchmarking.evaluate_qasper_f1 import read_answer_attempts, validate_gold_bundle
from benchmarking.qasper_judge_input import (
    ANNOTATION_EVIDENCE_POLICY,
    evidence_for_reference,
    semantic_evaluation_contract,
)


EVAL_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVAL_DIR / "runs"
DEFAULT_REPORT_NAME = "evaluation_report.md"
# Fixed presentation metadata, matching the public links in ui/branding.py.
PROJECT_REPOSITORY_URL = "https://github.com/filipe-braiman/measured-rag"
AUTHOR_GITHUB_URL = "https://github.com/filipe-braiman"
AUTHOR_LINKEDIN_URL = "https://linkedin.com/in/filipe-b-carvalho"
AUTHOR_EMAIL_URL = "mailto:filipebraiman@gmail.com"
REPORT_HEADER = (
    "# Measured RAG — QASPER Evaluation Report\n\n"
    "Documented end-to-end evaluation of the Measured RAG pipeline on a fixed QASPER-derived subset.\n\n"
    f"**Project repository:** [Measured RAG on GitHub]({PROJECT_REPOSITORY_URL})\n\n"
)
REPORT_FOOTER = (
    "\n---\n\n**Measured RAG**  \n"
    "Developed by **Filipe Braiman Carvalho**"
    f" · [GitHub]({AUTHOR_GITHUB_URL})"
    f" · [LinkedIn]({AUTHOR_LINKEDIN_URL})"
    f" · [Email]({AUTHOR_EMAIL_URL})\n"
)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
COMPLETED_JUDGE_STATUSES = {"complete", "complete_with_judge_errors"}
GENERATION_STATUSES = {"complete", "complete_with_errors"}
JUDGE_SCORE_STATUSES = {"success", "generation_error", "judge_error"}
RELIABILITY_EVALUATOR_VERSIONS = {
    "qasper-reference-geval-v1.2",
    "qasper-reference-geval-v1.3",
    "qasper-reference-geval-v1.4",
    "qasper-reference-geval-v1.5",
    "qasper-reference-geval-v1.6",
}
CURRENT_PIPELINE_VERSION = "qasper-eval-pipeline-v1.6"
V14_EVALUATOR_VERSION = "qasper-reference-geval-v1.4"
V15_EVALUATOR_VERSION = "qasper-reference-geval-v1.5"
V16_EVALUATOR_VERSION = "qasper-reference-geval-v1.6"
INDEPENDENT_REFERENCE_VERSIONS = {
    V14_EVALUATOR_VERSION,
    V15_EVALUATOR_VERSION,
    V16_EVALUATOR_VERSION,
}
QASPER_TYPES = ("extractive", "abstractive", "boolean", "none")
RELEVANCE_LABELS = ("high", "medium", "low", "unknown", "missing")
GROUNDEDNESS_LABELS = (
    "grounded",
    "partially_grounded",
    "not_grounded",
    "unknown",
    "missing",
)
CONFIG_FIELDS = (
    "profile_name",
    "profile_schema_version",
    "qa_data_path",
    "pdf_directory",
    "retrieval_mode",
    "chunk_size",
    "chunk_overlap",
    "retrieval_top_n",
    "fusion_threshold",
    "rerank_candidate_count",
    "rerank_top_k",
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
    "generation_case_interval_seconds",
)


class ReportValidationError(ValueError):
    """An artifact set is unsafe or internally inconsistent for reporting."""


def finite_number(value, label, minimum=None, maximum=None):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ReportValidationError(f"{label} must be a finite number.")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ReportValidationError(f"{label} must be at least {minimum}.")
    if maximum is not None and number > maximum:
        raise ReportValidationError(f"{label} must be at most {maximum}.")
    return number


def nonnegative_int(value, label):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ReportValidationError(f"{label} must be a nonnegative integer.")
    return value


def read_json(path, label, required=True):
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"{label} not found: {portable_path(path)}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"invalid numeric constant {value}")
                ),
            )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ReportValidationError(
            f"Malformed {label}: {portable_path(path)}"
        ) from exc
    if not isinstance(payload, dict):
        raise ReportValidationError(f"{label} must contain a JSON object.")
    return payload


def read_jsonl(path, label, required=True):
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"{label} not found: {portable_path(path)}")
        return None
    records = []
    line_number = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"invalid numeric constant {value}")
                    ),
                )
                if not isinstance(record, dict):
                    raise ValueError("record is not an object")
                records.append(record)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ReportValidationError(
            f"Malformed {label} at or before line {line_number}: "
            f"{portable_path(path)}"
        ) from exc
    if required and not records:
        raise ReportValidationError(f"{label} contains no records.")
    return records


def portable_path(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return "<external-path>"


def resolve_output_path(value, run_dir):
    if value is None:
        return run_dir / DEFAULT_REPORT_NAME
    if not isinstance(value, str) or not value.strip():
        raise ReportValidationError("Report output path must be non-empty.")
    if "\\" in value:
        raise ReportValidationError("Report output must use POSIX separators.")
    pure = PurePosixPath(value)
    if (
        not pure.parts
        or pure.is_absolute()
        or ".." in pure.parts
        or ":" in pure.parts[0]
    ):
        raise ReportValidationError(
            "Report output must be repository-relative and cannot traverse."
        )
    if pure.suffix.lower() != ".md":
        raise ReportValidationError("Report output must use a .md suffix.")
    resolved = (REPO_ROOT / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ReportValidationError(
            "Report output must remain inside the repository."
        ) from exc
    return resolved


def case_key(record):
    return str(record.get("paper_id")), str(record.get("question_id"))


def stage_status(pipeline_manifest, name):
    if not pipeline_manifest:
        return None
    stages = pipeline_manifest.get("stages")
    if not isinstance(stages, list):
        raise ReportValidationError("Pipeline manifest stages must be a list.")
    matches = [item for item in stages if item.get("stage") == name]
    if len(matches) > 1:
        raise ReportValidationError(f"Pipeline contains duplicate {name} stages.")
    return matches[0].get("status") if matches else None


def validate_run_id(container, field, run_id, label):
    if container.get(field) != run_id:
        raise ReportValidationError(f"{label} run ID does not match {run_id}.")


def validate_provenance(container, label):
    provenance = container.get("evaluation_profile")
    if provenance is None:
        return None
    if not isinstance(provenance, dict):
        raise ReportValidationError(f"{label} provenance must be an object.")
    configuration = provenance.get("resolved_configuration")
    digest = provenance.get("configuration_sha256")
    dataset_digest = provenance.get("dataset_sha256")
    if not isinstance(configuration, dict) or not isinstance(digest, str):
        raise ReportValidationError(
            f"{label} provenance lacks resolved configuration or digest."
        )
    calculated = hashlib.sha256(canonical_json(configuration)).hexdigest()
    if calculated != digest:
        raise ReportValidationError(f"{label} configuration digest is invalid.")
    if not isinstance(dataset_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", dataset_digest
    ):
        raise ReportValidationError(f"{label} dataset digest is invalid.")
    qa_path = configuration.get("qa_data_path")
    safe_qa_path = safe_recorded_path(qa_path)
    if safe_qa_path:
        local_path = REPO_ROOT / Path(*PurePosixPath(safe_qa_path).parts)
        if local_path.is_file() and sha256_file(local_path) != dataset_digest:
            raise ReportValidationError(
                f"{label} dataset digest does not match the recorded dataset."
            )
    return provenance


def safe_recorded_path(value):
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if not pure.parts:
        return None
    if not pure.is_absolute() and ":" not in pure.parts[0] and ".." not in pure.parts:
        return pure.as_posix()
    lowered = normalized.lower()
    for package in ("benchmarking", "eval"):
        marker = f"/{package}/"
        index = lowered.rfind(marker)
        if index >= 0:
            candidate = package + "/" + normalized[index + len(marker):]
            pure = PurePosixPath(candidate)
            if ".." not in pure.parts:
                return pure.as_posix()
    return None


def embedded_configuration(*containers):
    provenances = []
    for label, container in containers:
        if container:
            provenance = validate_provenance(container, label)
            if provenance:
                provenances.append((label, provenance))
    dataset_digests = {
        provenance["dataset_sha256"] for _, provenance in provenances
    }
    if len(dataset_digests) > 1:
        raise ReportValidationError("Artifact dataset digests disagree.")
    if provenances:
        baseline = provenances[0][1]["resolved_configuration"]
        judge_fields = {
            "profile_name",
            "judge_model", "judge_threshold", "judge_reasoning_effort",
            "judge_max_retries", "judge_max_completion_tokens",
            "judge_request_interval_seconds",
            "judge_retry_base_delay_seconds",
            "judge_retry_max_delay_seconds",
            "judge_retry_jitter_seconds",
            "judge_max_retry_after_seconds",
        }
        for label, provenance in provenances[1:]:
            candidate = provenance["resolved_configuration"]
            for field in CONFIG_FIELDS:
                legacy_case_pacing = (
                    field == "generation_case_interval_seconds"
                    and baseline.get(field) is None
                    and candidate.get(field) is not None
                )
                if (
                    field not in judge_fields
                    and not legacy_case_pacing
                    and candidate.get(field) != baseline.get(field)
                ):
                    raise ReportValidationError(
                        f"{label} generation/benchmark configuration disagrees."
                    )
    return provenances


def benchmark_configuration(generation_manifest, judge_manifest, provenances):
    generation_provenance = next(
        (item for label, item in provenances if label == "generation manifest"),
        None,
    )
    judge_provenance = next(
        (item for label, item in provenances if label == "judge manifest"),
        None,
    )
    if generation_provenance:
        resolved = dict(generation_provenance["resolved_configuration"])
        profile_name = generation_provenance.get("profile_name")
        config_digest = generation_provenance["configuration_sha256"]
        dataset_digest = generation_provenance["dataset_sha256"]
    else:
        legacy = generation_manifest.get("configuration")
        legacy = legacy if isinstance(legacy, dict) else {}
        paths = generation_manifest.get("paths")
        paths = paths if isinstance(paths, dict) else {}
        resolved = {
            "qa_data_path": safe_recorded_path(paths.get("qa_data")),
            "pdf_directory": safe_recorded_path(paths.get("pdf_directory")),
            "retrieval_mode": legacy.get("retrieval_mode"),
            "chunk_size": legacy.get("chunk_size"),
            "chunk_overlap": legacy.get("chunk_overlap"),
            "retrieval_top_n": legacy.get("retrieval_top_n"),
            "fusion_threshold": legacy.get("fusion_threshold"),
            "rerank_candidate_count": legacy.get("rerank_candidate_count", legacy.get("rerank_candidate_n")),
            "rerank_top_k": legacy.get("rerank_top_k"),
        }
        profile_name = None
        config_digest = None
        dataset_digest = None

    if judge_provenance:
        effective = judge_provenance["resolved_configuration"]
        for field in (
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
        ):
            resolved[field] = effective.get(field)
    elif judge_manifest:
        judge_request = judge_manifest.get("judge_request")
        if not isinstance(judge_request, dict):
            judge_request = judge_manifest.get("configuration")
        judge_request = judge_request if isinstance(judge_request, dict) else {}
        resolved.update(
            {
                "judge_model": judge_manifest.get("judge_model"),
                "judge_threshold": judge_request.get("threshold"),
                "judge_reasoning_effort": judge_request.get("reasoning_effort"),
                "judge_max_retries": judge_request.get("max_retries"),
                "judge_max_completion_tokens": judge_request.get("max_completion_tokens"),
                "judge_request_interval_seconds": judge_request.get("request_interval_seconds"),
                "judge_retry_base_delay_seconds": judge_request.get("retry_base_delay_seconds"),
                "judge_retry_max_delay_seconds": judge_request.get("retry_max_delay_seconds"),
                "judge_retry_jitter_seconds": judge_request.get("retry_jitter_seconds"),
                "judge_max_retry_after_seconds": judge_request.get("max_retry_after_seconds"),
            }
        )
    return {
        "resolved": resolved,
        "profile_name": profile_name,
        "configuration_sha256": config_digest,
        "dataset_sha256": dataset_digest,
    }


def validate_generation(run_id, manifest_path, answers_path):
    manifest = read_json(manifest_path, "generation manifest")
    validate_run_id(manifest, "run_id", run_id, "Generation manifest")
    if manifest.get("status") not in GENERATION_STATUSES:
        raise ReportValidationError("Generation run is not complete.")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ReportValidationError("Generation manifest counts are missing.")
    selected_count = nonnegative_int(
        counts.get("selected_questions"), "generation selected questions"
    )
    nonnegative_int(counts.get("selected_papers"), "generation selected papers")
    if selected_count < 1:
        raise ReportValidationError("Generation selected no benchmark cases.")
    try:
        answers, duplicate_attempts = read_answer_attempts(answers_path)
    except (OSError, ValueError, TypeError) as exc:
        raise ReportValidationError(f"Invalid generation answers: {exc}") from exc
    if len(answers) != selected_count:
        raise ReportValidationError(
            "Generation selected-case count does not match unique answers."
        )
    keys = [case_key(record) for record in answers]
    if len(keys) != len(set(keys)):
        raise ReportValidationError("Selected generation question keys are not unique.")
    selection = manifest.get("case_selection")
    if selection is not None:
        serialized_keys = [f"{paper_id}/{question_id}" for paper_id, question_id in keys]
        dataset_digest = (manifest.get("evaluation_profile") or {}).get(
            "dataset_sha256"
        )
        if (
            not isinstance(selection, dict)
            or selection.get("selection_mode")
            != "prior_semantic_below_threshold"
            or selection.get("selected_case_count") != selected_count
            or selection.get("ordered_case_keys") != serialized_keys
            or selection.get("ordered_case_keys_sha256")
            != sha256_json(serialized_keys)
            or selection.get("source_dataset_sha256") != dataset_digest
            or not isinstance(selection.get("source_run_id"), str)
            or not isinstance(selection.get("source_judge_id"), str)
            or not isinstance(selection.get("source_evaluator_version"), str)
        ):
            raise ReportValidationError(
                "Generation failure-subset selection provenance is invalid."
            )
        finite_number(
            selection.get("source_threshold"),
            "source semantic threshold",
            0.0,
            1.0,
        )
    for record in answers:
        finite_number(record.get("elapsed_seconds"), "answer elapsed time", 0.0)
    successful = sum(record.get("status") == "success" for record in answers)
    errors = selected_count - successful
    recorded_success = counts.get("successful")
    recorded_failed = counts.get("failed")
    if isinstance(recorded_success, int) and recorded_success != successful:
        raise ReportValidationError("Generation successful count is contradictory.")
    if isinstance(recorded_failed, int) and recorded_failed != errors:
        raise ReportValidationError("Generation error count is contradictory.")
    return manifest, answers, duplicate_attempts, successful, errors


def validate_qasper(
    run_id, metrics_path, scores_path, answers, duplicate_generation_attempts=0
):
    metrics = read_json(metrics_path, "QASPER metrics")
    scores = read_jsonl(scores_path, "QASPER scores")
    validate_run_id(metrics, "run_id", run_id, "QASPER metrics")
    answer_keys = [case_key(item) for item in answers]
    score_keys = [case_key(item) for item in scores]
    if len(score_keys) != len(set(score_keys)) or score_keys != answer_keys:
        raise ReportValidationError(
            "QASPER scores must contain exactly one result per selected case "
            "in benchmark order."
        )
    selected = len(answers)
    for field in ("selected_question_count", "unique_evaluated_case_count"):
        if metrics.get(field) != selected:
            raise ReportValidationError(f"QASPER {field} is contradictory.")
    values = []
    by_type = defaultdict(list)
    generation_errors = 0
    for answer, score in zip(answers, scores):
        if score.get("question") != answer.get("question"):
            raise ReportValidationError("QASPER score question text changed.")
        value = finite_number(score.get("answer_f1"), "QASPER case F1", 0.0, 1.0)
        answer_type = score.get("matched_answer_type")
        if answer_type not in QASPER_TYPES:
            raise ReportValidationError("QASPER score has an invalid answer type.")
        if answer.get("status") == "error":
            generation_errors += 1
            if value != 0.0:
                raise ReportValidationError("Generation errors must have zero QASPER F1.")
        values.append(value)
        by_type[answer_type].append(value)
    overall = finite_number(metrics.get("answer_f1"), "QASPER Answer F1", 0.0, 1.0)
    percentage = finite_number(
        metrics.get("answer_f1_percentage"), "QASPER Answer F1 percentage", 0.0, 100.0
    )
    calculated = statistics.fmean(values)
    if not math.isclose(overall, calculated, rel_tol=0.0, abs_tol=1e-12):
        raise ReportValidationError("QASPER Answer F1 does not match case scores.")
    if not math.isclose(percentage, overall * 100, rel_tol=0.0, abs_tol=1e-10):
        raise ReportValidationError("QASPER percentage does not match Answer F1.")
    if metrics.get("generation_error_count") != generation_errors:
        raise ReportValidationError("QASPER generation-error count is contradictory.")
    if metrics.get("successful_prediction_count") != selected - generation_errors:
        raise ReportValidationError("QASPER successful-prediction count is contradictory.")
    if metrics.get("duplicate_attempt_count") != duplicate_generation_attempts:
        raise ReportValidationError("QASPER duplicate-attempt count is contradictory.")
    recorded_by_type = metrics.get("answer_f1_by_matched_answer_type")
    if not isinstance(recorded_by_type, dict):
        raise ReportValidationError("QASPER answer-type metrics are missing.")
    for answer_type in QASPER_TYPES:
        expected = statistics.fmean(by_type[answer_type]) if by_type[answer_type] else 0.0
        actual = finite_number(
            recorded_by_type.get(answer_type), f"QASPER {answer_type} F1", 0.0, 1.0
        )
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ReportValidationError(f"QASPER {answer_type} F1 is contradictory.")
    return metrics, scores


def judge_resume_key(record):
    return json.dumps(
        [
            record.get("paper_id"),
            record.get("question_id"),
            record.get("judge_model"),
            record.get("evaluator_version"),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def select_latest_judge_scores(records, judge_model, evaluator_version):
    attempts = defaultdict(list)
    for line_number, record in enumerate(records, start=1):
        required = {
            "paper_id", "question_id", "judge_model", "evaluator_version",
            "resume_key", "status", "attempt_number",
        }
        missing = required - record.keys()
        if missing:
            raise ReportValidationError(
                f"Judge score line {line_number} is missing: " + ", ".join(sorted(missing))
            )
        if record["judge_model"] != judge_model or record["evaluator_version"] != evaluator_version:
            raise ReportValidationError("Judge score identity does not match its manifest.")
        expected_key = judge_resume_key(record)
        if record["resume_key"] != expected_key:
            raise ReportValidationError("Judge score has an invalid resume key.")
        if record["status"] not in JUDGE_SCORE_STATUSES:
            raise ReportValidationError("Judge score has an invalid status.")
        attempt = record["attempt_number"]
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise ReportValidationError("Judge score has an invalid attempt number.")
        attempts[expected_key].append(record)
    selected = {}
    for key, group in attempts.items():
        successful = [item for item in group if item["status"] == "success"]
        generation_errors = [item for item in group if item["status"] == "generation_error"]
        selected[key] = (
            successful[-1]
            if successful
            else generation_errors[-1]
            if generation_errors
            else group[-1]
        )
    return attempts, selected


def validate_judge_transport(record, label):
    telemetry = record.get("judge_transport")
    if not isinstance(telemetry, dict):
        raise ReportValidationError(f"{label} lacks judge transport telemetry.")
    count_fields = (
        "provider_attempt_count", "retry_count", "rate_limit_response_count",
        "rate_limit_retry_count", "other_transient_retry_count",
        "schema_output_retry_count",
    )
    for field in count_fields:
        nonnegative_int(telemetry.get(field), f"{label} {field}")
    for field in ("pacing_sleep_seconds", "retry_backoff_sleep_seconds"):
        finite_number(telemetry.get(field), f"{label} {field}", 0.0)
    if not isinstance(telemetry.get("retry_after_used"), bool):
        raise ReportValidationError(f"{label} has invalid retry_after_used.")
    if (
        telemetry["rate_limit_retry_count"]
        + telemetry["other_transient_retry_count"]
        + telemetry["schema_output_retry_count"]
        != telemetry["retry_count"]
    ):
        raise ReportValidationError(f"{label} retry categories do not reconcile.")
    if telemetry["retry_count"] > telemetry["provider_attempt_count"]:
        raise ReportValidationError(f"{label} retries exceed provider attempts.")
    if telemetry["rate_limit_retry_count"] > telemetry["rate_limit_response_count"]:
        raise ReportValidationError(f"{label} rate-limit retries exceed responses.")
    if telemetry["retry_after_used"] and telemetry["rate_limit_retry_count"] == 0:
        raise ReportValidationError(
            f"{label} used retry-after without a rate-limit retry."
        )
    return telemetry


def judge_reliability_aggregates(selected, all_attempts):
    telemetry = [
        validate_judge_transport(record, "Selected judge score")
        for record in selected
    ]
    return {
        "total_provider_attempts": sum(item["provider_attempt_count"] for item in telemetry),
        "cases_requiring_retry_count": sum(item["retry_count"] > 0 for item in telemetry),
        "total_retry_count": sum(item["retry_count"] for item in telemetry),
        "total_rate_limit_response_count": sum(item["rate_limit_response_count"] for item in telemetry),
        "total_rate_limit_retry_count": sum(item["rate_limit_retry_count"] for item in telemetry),
        "total_other_transient_retry_count": sum(item["other_transient_retry_count"] for item in telemetry),
        "total_schema_output_retry_count": sum(item["schema_output_retry_count"] for item in telemetry),
        "total_pacing_sleep_seconds": round(sum(item["pacing_sleep_seconds"] for item in telemetry), 6),
        "total_retry_backoff_sleep_seconds": round(sum(item["retry_backoff_sleep_seconds"] for item in telemetry), 6),
        "telemetry_covered_selected_case_count": len(telemetry),
        "unresolved_judge_error_count": sum(record.get("status") == "judge_error" for record in selected),
        "historical_judge_error_attempt_count": sum(record.get("status") == "judge_error" for records in all_attempts.values() for record in records),
    }


def validate_v14_reference_record(
    score,
    answer,
    threshold,
    label,
    require_annotation_evidence=False,
):
    references = score.get("references")
    expected = validate_gold_bundle(
        answer["gold_answers"],
        f"{answer['paper_id']}/{answer['question_id']}",
    )
    if references != expected:
        raise ReportValidationError(f"{label} references contradict generation.")
    evaluations = score.get("reference_evaluations")
    if not isinstance(evaluations, list):
        raise ReportValidationError(f"{label} reference evaluations are missing.")
    if score["status"] == "generation_error":
        if evaluations:
            raise ReportValidationError(
                f"{label} generation error invoked reference judgments."
            )
        return len(references), 0, 0, None, False
    if len(evaluations) != len(references):
        raise ReportValidationError(
            f"{label} lacks one judgment per valid annotation."
        )

    successes = []
    errors = []
    summed_attempts = 0
    for index, (reference, evaluation) in enumerate(zip(references, evaluations)):
        identity = {
            "reference_index": index,
            "annotation_id": reference["annotation_id"],
            "answer_type": reference["answer_type"],
            "reference_answer": reference["answer"],
        }
        if any(evaluation.get(field) != value for field, value in identity.items()):
            raise ReportValidationError(f"{label} reference {index} is misaligned.")
        if require_annotation_evidence:
            expected_evidence = evidence_for_reference(
                answer["gold_answers"], reference
            )
            if evaluation.get("judge_evidence") != expected_evidence:
                raise ReportValidationError(
                    f"{label} reference {index} has incorrect annotation evidence."
                )
        telemetry = validate_judge_transport(
            evaluation, f"{label} reference {index}"
        )
        summed_attempts += telemetry["provider_attempt_count"]
        if evaluation.get("status") == "success":
            value = finite_number(
                evaluation.get("semantic_score"),
                f"{label} reference {index} score",
                0.0,
                1.0,
            )
            if (
                evaluation.get("passed") is not (value >= threshold)
                or not isinstance(evaluation.get("judge_reason"), str)
                or evaluation.get("judge_error") is not None
            ):
                raise ReportValidationError(
                    f"{label} successful reference {index} is malformed."
                )
            successes.append(evaluation)
        elif evaluation.get("status") == "judge_error":
            if (
                evaluation.get("semantic_score") is not None
                or evaluation.get("passed") is not None
                or evaluation.get("judge_reason") is not None
                or not isinstance(evaluation.get("judge_error"), dict)
            ):
                raise ReportValidationError(
                    f"{label} failed reference {index} is malformed."
                )
            errors.append(evaluation)
        else:
            raise ReportValidationError(f"{label} reference {index} has invalid status.")
    if validate_judge_transport(score, label)["provider_attempt_count"] != summed_attempts:
        raise ReportValidationError(
            f"{label} provider attempts do not equal reference-level attempts."
        )
    if errors:
        if (
            score["status"] != "judge_error"
            or score.get("semantic_score") is not None
            or score.get("passed") is not None
            or any(
                score.get(field) is not None
                for field in (
                    "matched_reference_index",
                    "matched_annotation_id",
                    "matched_reference",
                    "matched_answer_type",
                )
            )
        ):
            raise ReportValidationError(
                f"{label} incomplete reference coverage is invalid."
            )
        return len(references), len(successes), len(errors), None, False
    winner = max(successes, key=lambda item: item["semantic_score"])
    expected_winner = {
        "matched_reference_index": winner["reference_index"],
        "matched_annotation_id": winner["annotation_id"],
        "matched_reference": winner["reference_answer"],
        "matched_answer_type": winner["answer_type"],
    }
    if any(score.get(field) != value for field, value in expected_winner.items()):
        raise ReportValidationError(f"{label} maximum-reference selection is invalid.")
    if (
        score.get("semantic_score") != winner["semantic_score"]
        or score.get("passed") is not (winner["semantic_score"] >= threshold)
        or score.get("judge_reason") != winner["judge_reason"]
    ):
        raise ReportValidationError(f"{label} winner fields are contradictory.")
    return (
        len(references), len(successes), 0, winner["answer_type"],
        winner["reference_index"] != 0,
    )


def validate_deepeval(
    run_id,
    judge_id,
    manifest_path,
    metrics_path,
    scores_path,
    answers,
    case_selection=None,
):
    manifest = read_json(manifest_path, "judge manifest")
    metrics = read_json(metrics_path, "DeepEval metrics")
    records = read_jsonl(scores_path, "DeepEval scores")
    validate_run_id(manifest, "generation_run_id", run_id, "Judge manifest")
    validate_run_id(metrics, "generation_run_id", run_id, "DeepEval metrics")
    if manifest.get("judge_id") != judge_id or metrics.get("judge_id") != judge_id:
        raise ReportValidationError("DeepEval judge IDs disagree.")
    if manifest.get("status") not in COMPLETED_JUDGE_STATUSES:
        raise ReportValidationError("Selected judge run is not complete.")
    judge_model = manifest.get("judge_model")
    evaluator_version = manifest.get("evaluator_version")
    if metrics.get("judge_model") != judge_model or metrics.get("evaluator_version") != evaluator_version:
        raise ReportValidationError("DeepEval model or evaluator version disagrees.")
    if (
        manifest.get("case_selection") != case_selection
        or metrics.get("case_selection") != case_selection
    ):
        raise ReportValidationError(
            "DeepEval case-selection provenance contradicts generation."
        )
    if evaluator_version in {V15_EVALUATOR_VERSION, V16_EVALUATOR_VERSION}:
        policy = manifest.get("qasper_placeholder_policy")
        if (
            not isinstance(policy, dict)
            or metrics.get("qasper_placeholder_policy") != policy
            or policy.get("policy_version") != "qasper-placeholder-markup-v1"
            or policy.get("reference_transformation") != "none"
            or policy.get("original_converted_reference_preserved") is not True
        ):
            raise ReportValidationError(
                "DeepEval QASPER placeholder provenance is invalid."
            )
    if evaluator_version == V16_EVALUATOR_VERSION:
        evidence_policy = manifest.get("qasper_annotation_evidence_policy")
        if (
            evidence_policy != ANNOTATION_EVIDENCE_POLICY
            or metrics.get("qasper_annotation_evidence_policy")
            != evidence_policy
            or manifest.get("evaluation_steps") != metrics.get("evaluation_steps")
            or manifest.get("rubric") != metrics.get("rubric")
            or manifest.get("reference_semantics_note")
            != metrics.get("reference_semantics_note")
            or "absence from an abbreviated reference is not itself evidence of fabrication"
            not in str(manifest.get("reference_semantics_note", "")).lower()
        ):
            raise ReportValidationError(
                "DeepEval annotation-evidence provenance is invalid."
            )
    attempts, selected_by_resume = select_latest_judge_scores(
        records, judge_model, evaluator_version
    )
    selected_records = list(selected_by_resume.values())
    selected_count = nonnegative_int(
        metrics.get("selected_case_count"), "DeepEval selected-case count"
    )
    if selected_count > len(answers) or len(selected_records) != selected_count:
        raise ReportValidationError("DeepEval selected-case count is contradictory.")
    expected_keys = [case_key(item) for item in answers[:selected_count]]
    selected_by_case = {case_key(item): item for item in selected_records}
    if len(selected_by_case) != len(selected_records) or set(selected_by_case) != set(expected_keys):
        raise ReportValidationError("DeepEval cases do not match the benchmark prefix.")
    ordered = [selected_by_case[key] for key in expected_keys]
    threshold = finite_number(metrics.get("threshold"), "DeepEval threshold", 0.0, 1.0)
    semantic_values = []
    success_count = generation_error_count = judge_error_count = 0
    reference_distribution = Counter()
    matched_types = Counter({kind: 0 for kind in QASPER_TYPES})
    expected_reference_count = successful_reference_count = reference_error_count = 0
    non_first_winner_count = 0
    for case_index, (answer, score) in enumerate(
        zip(answers[:selected_count], ordered), start=1
    ):
        status = score["status"]
        if status == "success":
            success_count += 1
            value = finite_number(score.get("semantic_score"), "semantic score", 0.0, 1.0)
            if not isinstance(score.get("passed"), bool) or score["passed"] != (value >= threshold):
                raise ReportValidationError("Semantic pass value is contradictory.")
            if answer.get("status") != "success":
                raise ReportValidationError("A generation error cannot have a successful judgment.")
            semantic_values.append(value)
        elif status == "generation_error":
            generation_error_count += 1
            value = finite_number(score.get("semantic_score"), "generation-error semantic score", 0.0, 1.0)
            if value != 0.0 or score.get("passed") is not False or answer.get("status") != "error":
                raise ReportValidationError("Generation-error semantic handling is invalid.")
            semantic_values.append(value)
        else:
            judge_error_count += 1
            if score.get("semantic_score") is not None or score.get("passed") is not None:
                raise ReportValidationError("Judge errors must retain null semantic results.")
            if answer.get("status") != "success":
                raise ReportValidationError("Generation errors must not become judge errors.")
        finite_number(score.get("elapsed_seconds"), "judge elapsed time", 0.0)
        if evaluator_version in INDEPENDENT_REFERENCE_VERSIONS:
            (
                reference_count,
                reference_successes,
                reference_errors,
                matched_type,
                non_first,
            ) = validate_v14_reference_record(
                score,
                answer,
                threshold,
                f"Selected semantic case {case_index}",
                require_annotation_evidence=(
                    evaluator_version == V16_EVALUATOR_VERSION
                ),
            )
            reference_distribution[str(reference_count)] += 1
            if answer["status"] == "success":
                expected_reference_count += reference_count
            successful_reference_count += reference_successes
            reference_error_count += reference_errors
            non_first_winner_count += non_first
            if matched_type is not None:
                matched_types[matched_type] += 1
    non_null = len(semantic_values)
    if success_count + generation_error_count + judge_error_count != selected_count:
        raise ReportValidationError("DeepEval result counts do not reconcile.")
    expected_metrics = {
        "successful_judgment_count": success_count,
        "generation_error_count": generation_error_count,
        "judge_error_count": judge_error_count,
        "non_null_evaluated_case_count": non_null,
    }
    for field, expected in expected_metrics.items():
        if metrics.get(field) != expected:
            raise ReportValidationError(f"DeepEval {field} is contradictory.")
    if evaluator_version in INDEPENDENT_REFERENCE_VERSIONS:
        reference_metrics = {
            "expected_reference_judgment_count": expected_reference_count,
            "successful_reference_judgment_count": successful_reference_count,
            "reference_judgment_error_count": reference_error_count,
            "reference_count_distribution": dict(sorted(reference_distribution.items())),
            "cases_matched_by_answer_type": dict(matched_types),
            "non_first_winning_reference_count": non_first_winner_count,
        }
        for field, expected in reference_metrics.items():
            if metrics.get(field) != expected:
                raise ReportValidationError(
                    f"DeepEval reference aggregate {field} is contradictory."
                )
    coverage = finite_number(metrics.get("evaluation_coverage"), "semantic coverage", 0.0, 1.0)
    expected_coverage = non_null / selected_count if selected_count else 0.0
    if not math.isclose(coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12):
        raise ReportValidationError("DeepEval coverage is contradictory.")
    mean = metrics.get("mean_semantic_correctness")
    pass_rate = metrics.get("pass_rate")
    if non_null:
        mean = finite_number(mean, "mean semantic correctness", 0.0, 1.0)
        pass_rate = finite_number(pass_rate, "semantic pass rate", 0.0, 1.0)
        expected_mean = statistics.fmean(semantic_values)
        expected_pass = sum(item >= threshold for item in semantic_values) / non_null
        if not math.isclose(mean, expected_mean, rel_tol=0.0, abs_tol=1e-12):
            raise ReportValidationError("Semantic mean is contradictory.")
        if not math.isclose(pass_rate, expected_pass, rel_tol=0.0, abs_tol=1e-12):
            raise ReportValidationError("Semantic pass rate is contradictory.")
    elif mean is not None or pass_rate is not None:
        raise ReportValidationError("Zero semantic coverage requires null mean and pass rate.")
    full_run = metrics.get("full_run_semantic_correctness")
    if coverage == 1.0:
        if case_selection is not None:
            if full_run is not None:
                raise ReportValidationError(
                    "A failure subset cannot report full-run semantic correctness."
                )
        else:
            full_run = finite_number(full_run, "full-run semantic correctness", 0.0, 1.0)
            if not math.isclose(full_run, mean, rel_tol=0.0, abs_tol=1e-12):
                raise ReportValidationError("Full-run semantic correctness is contradictory.")
    elif full_run is not None:
        raise ReportValidationError("Partial semantic coverage requires a null full-run score.")
    rubric_counts = metrics.get("score_counts_by_rubric_band")
    if not isinstance(rubric_counts, dict):
        raise ReportValidationError("DeepEval rubric-band distribution is missing.")
    for band, count in rubric_counts.items():
        nonnegative_int(count, f"DeepEval rubric band {band}")
    if sum(rubric_counts.values()) != non_null:
        raise ReportValidationError("DeepEval rubric-band counts are contradictory.")
    manifest_counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    historical_errors = sum(
        item.get("status") == "judge_error" for group in attempts.values() for item in group
    )
    if "judge_error_attempt_count" in manifest_counts and manifest_counts["judge_error_attempt_count"] != historical_errors:
        raise ReportValidationError("Historical judge-error attempt count is contradictory.")
    if "unresolved_judge_error_count" in manifest_counts and manifest_counts["unresolved_judge_error_count"] != judge_error_count:
        raise ReportValidationError("Unresolved judge-error count is contradictory.")
    if evaluator_version in INDEPENDENT_REFERENCE_VERSIONS:
        for field in (
            "expected_reference_judgment_count",
            "successful_reference_judgment_count",
            "reference_judgment_error_count",
            "reference_count_distribution",
            "cases_matched_by_answer_type",
            "non_first_winning_reference_count",
        ):
            if manifest_counts.get(field) != metrics.get(field):
                raise ReportValidationError(
                    f"Judge manifest reference aggregate {field} is contradictory."
                )
    if evaluator_version in RELIABILITY_EVALUATOR_VERSIONS:
        reliability = judge_reliability_aggregates(ordered, attempts)
        recorded = metrics.get("transport_reliability")
        if not isinstance(recorded, dict):
            raise ReportValidationError("Judge reliability metrics are missing.")
        for field, expected in reliability.items():
            actual = recorded.get(field)
            if isinstance(expected, float):
                actual = finite_number(actual, f"judge reliability {field}", 0.0)
                if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6):
                    raise ReportValidationError(
                        f"Judge reliability {field} is contradictory."
                    )
            elif actual != expected:
                raise ReportValidationError(
                    f"Judge reliability {field} is contradictory."
                )
    return manifest, metrics, ordered, attempts


def select_judge_id(run_dir, explicit, pipeline_manifest, pipeline_summary):
    if explicit:
        if not RUN_ID_RE.fullmatch(explicit):
            raise ReportValidationError("Invalid judge ID.")
        return explicit
    recorded = []
    if pipeline_summary:
        value = (pipeline_summary.get("deepeval") or {}).get("judge_id")
        if value:
            recorded.append(value)
    if pipeline_manifest:
        value = (pipeline_manifest.get("invocation_configuration") or {}).get("judge_id")
        if value and value not in recorded:
            recorded.append(value)
    if len(recorded) > 1:
        raise ReportValidationError(
            "Pipeline artifacts disagree on judge ID: " + ", ".join(recorded)
        )
    if recorded:
        judge_id = recorded[0]
        judge_manifest = read_json(
            run_dir / "deepeval" / judge_id / "judge_manifest.json",
            "recorded judge manifest",
            required=False,
        )
        if judge_manifest and judge_manifest.get("status") in COMPLETED_JUDGE_STATUSES:
            return judge_id
        raise ReportValidationError(
            f"Pipeline-recorded judge {judge_id!r} is not a valid completed run."
        )
    candidates = []
    deepeval_dir = run_dir / "deepeval"
    if deepeval_dir.is_dir():
        for directory in sorted(item for item in deepeval_dir.iterdir() if item.is_dir()):
            manifest = read_json(directory / "judge_manifest.json", "judge manifest", required=False)
            metrics = directory / "metrics.json"
            scores = directory / "scores.jsonl"
            if (
                manifest
                and manifest.get("status") in COMPLETED_JUDGE_STATUSES
                and metrics.is_file()
                and scores.is_file()
            ):
                candidates.append(directory.name)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    raise ReportValidationError(
        "Multiple completed judge runs are ambiguous; specify --judge-id. "
        "Candidates: " + ", ".join(f"deepeval/{item}" for item in candidates)
    )


def pipeline_recorded_judge_id(pipeline_manifest, pipeline_summary):
    recorded = []
    if pipeline_summary:
        value = (pipeline_summary.get("deepeval") or {}).get("judge_id")
        if value:
            recorded.append(value)
    if pipeline_manifest:
        value = (pipeline_manifest.get("invocation_configuration") or {}).get(
            "judge_id"
        )
        if value and value not in recorded:
            recorded.append(value)
    if len(recorded) > 1:
        raise ReportValidationError(
            "Pipeline artifacts disagree on judge ID: " + ", ".join(recorded)
        )
    return recorded[0] if recorded else None


def validate_matching_pipeline_judge(
    selected_judge_id,
    pipeline_judge_id,
    pipeline_manifest,
    pipeline_summary,
    metrics,
):
    if not metrics or selected_judge_id != pipeline_judge_id:
        return
    semantic_summary = (pipeline_summary or {}).get("deepeval") or {}
    comparisons = {
        "judge_id": selected_judge_id,
        "model": metrics.get("judge_model"),
        "evaluator_version": metrics.get("evaluator_version"),
        "successful_judgments": metrics.get("successful_judgment_count"),
        "generation_errors": metrics.get("generation_error_count"),
        "judge_errors": metrics.get("judge_error_count"),
        "evaluation_coverage": metrics.get("evaluation_coverage"),
        "mean_semantic_correctness": metrics.get("mean_semantic_correctness"),
        "pass_rate": metrics.get("pass_rate"),
        "full_run_semantic_correctness": metrics.get(
            "full_run_semantic_correctness"
        ),
    }
    for summary_field, expected in comparisons.items():
        actual = semantic_summary.get(summary_field)
        if actual is not None and actual != expected:
            raise ReportValidationError(
                f"Pipeline semantic {summary_field} is contradictory."
            )
    manifest_model = (pipeline_manifest or {}).get("models", {}).get("judge")
    if manifest_model is not None and manifest_model != metrics.get("judge_model"):
        raise ReportValidationError("Pipeline semantic model is contradictory.")


def normalize_diagnostic_label(value, allowed):
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    for label in sorted(allowed, key=len, reverse=True):
        if normalized == label or normalized.endswith("_" + label):
            return label
    return "unknown"


def validate_application_logs(path, answers):
    records = read_jsonl(path, "application logs", required=False)
    if records is None:
        return None
    expected = {record["conversation_id"]: case_key(record) for record in answers}
    matched = defaultdict(list)
    for record in records:
        conversation_id = record.get("conversation_id")
        if conversation_id in expected:
            matched[conversation_id].append(record)
    missing = [conversation_id for conversation_id in expected if not matched[conversation_id]]
    duplicates = [
        conversation_id for conversation_id in expected if len(matched[conversation_id]) > 1
    ]
    by_case = {}
    for conversation_id, key in expected.items():
        if len(matched[conversation_id]) == 1:
            by_case[key] = matched[conversation_id][0]
        else:
            by_case[key] = None
    return {
        "records": records,
        "by_case": by_case,
        "matched_case_count": sum(value is not None for value in by_case.values()),
        "missing_conversation_ids": missing,
        "duplicate_conversation_ids": duplicates,
    }


def diagnostic_labels(answers, application_logs):
    relevance = Counter({label: 0 for label in RELEVANCE_LABELS})
    groundedness = Counter({label: 0 for label in GROUNDEDNESS_LABELS})
    by_case = {}
    for answer in answers:
        key = case_key(answer)
        log = application_logs["by_case"].get(key) if application_logs else None
        if log is None:
            relevance_label = "missing"
            groundedness_label = "missing"
        else:
            retrieval = log.get("retrieval")
            generation = log.get("generation")
            retrieval = retrieval if isinstance(retrieval, dict) else {}
            generation = generation if isinstance(generation, dict) else {}
            relevance_label = normalize_diagnostic_label(
                retrieval.get("retrieval_relevance"), RELEVANCE_LABELS[:-1]
            )
            groundedness_label = normalize_diagnostic_label(
                generation.get("groundedness_label"), GROUNDEDNESS_LABELS[:-1]
            )
        relevance[relevance_label] += 1
        groundedness[groundedness_label] += 1
        by_case[key] = {
            "relevance": relevance_label,
            "groundedness": groundedness_label,
        }
    return relevance, groundedness, by_case


def numeric_series(values, label):
    series = []
    for value in values:
        if value is None:
            continue
        series.append(finite_number(value, label, 0.0))
    return series


def nested_value(container, *keys):
    current = container
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def latency_and_usage(answers, application_logs, judge_scores):
    logs = (
        [record for record in application_logs["by_case"].values() if record]
        if application_logs
        else []
    )
    latency = {
        "End-to-end answer": numeric_series(
            (item.get("elapsed_seconds") for item in answers), "answer latency"
        ),
        "Total retrieval": numeric_series(
            (nested_value(item, "retrieval", "latency", "total_seconds") for item in logs),
            "retrieval latency",
        ),
        "Dense retrieval": numeric_series(
            (nested_value(item, "retrieval", "latency", "dense_seconds") for item in logs),
            "dense latency",
        ),
        "BM25": numeric_series(
            (nested_value(item, "retrieval", "latency", "bm25_seconds") for item in logs),
            "BM25 latency",
        ),
        "Fusion": numeric_series(
            (nested_value(item, "retrieval", "latency", "fusion_seconds") for item in logs),
            "fusion latency",
        ),
        "Reranking": numeric_series(
            (nested_value(item, "retrieval", "latency", "rerank_seconds") for item in logs),
            "reranking latency",
        ),
        "Generator": numeric_series(
            (nested_value(item, "llm_metrics", "latency_seconds") for item in logs),
            "generator latency",
        ),
        "Relevance checker": numeric_series(
            (nested_value(item, "retrieval", "retrieval_eval_latency") for item in logs),
            "relevance checker latency",
        ),
        "Groundedness checker": numeric_series(
            (nested_value(item, "generation", "groundedness_eval_latency") for item in logs),
            "groundedness checker latency",
        ),
        "Semantic judge": numeric_series(
            (item.get("elapsed_seconds") for item in (judge_scores or [])),
            "semantic judge latency",
        ),
    }
    tokens = numeric_series(
        (nested_value(item, "llm_metrics", "total_tokens") for item in logs),
        "generation token count",
    )
    return latency, tokens


def percentile_nearest_rank(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def series_summary(values):
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile_nearest_rank(values, 0.95),
        "maximum": max(values),
    }


def format_percentage(value, digits=2):
    return "N/A" if value is None else f"{value * 100:.{digits}f}%"


def format_ratio(count, total):
    percentage = count / total if total else 0.0
    return f"{count}/{total} ({percentage * 100:.1f}%)"


def format_number(value, digits=4):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def recorded_token_usage_lines(tokens):
    """Format aggregates over available application-log token observations."""
    if not tokens:
        return [
            "- Total recorded generation tokens: **Not available**.",
            "- Mean generation tokens per recorded case: **Not available**.",
        ]
    token_total = sum(tokens)
    token_mean = statistics.fmean(tokens)
    return [
        f"- Total recorded generation tokens (n={len(tokens)}): **{format_number(token_total, 0)}**.",
        f"- Mean generation tokens per recorded case (n={len(tokens)}): **{format_number(token_mean, 2)}**.",
    ]


def display_value(value):
    if value is None or value == "":
        return "Not recorded"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def markdown_text(value, limit=None):
    if value is None:
        return "N/A"
    text = " ".join(str(value).split())
    if limit is not None and len(text) > limit:
        text = text[: max(0, limit - 1)].rstrip() + "…"
    text = html.escape(text, quote=False)
    return text.replace("|", "\\|").replace("`", "&#96;")


def short_question_id(value):
    value = str(value)
    return value if len(value) <= 12 else value[:12] + "…"


def relative_link(report_path, target):
    relative = os.path.relpath(Path(target), start=Path(report_path).parent)
    return relative.replace("\\", "/")


def artifact_link(report_path, target, label):
    path = Path(target)
    portable = portable_path(path)
    if not path.is_file():
        return f"{label}: `{portable}` (not available)"
    return f"{label}: [`{portable}`]({relative_link(report_path, path)})"


def extract_models(generation_manifest, deepeval_metrics):
    models = generation_manifest.get("models")
    models = models if isinstance(models, dict) else {}
    shared_checker = models.get("checker")
    return {
        "Embedding": models.get("embedding"),
        "Reranker": models.get("reranker"),
        "Generator": models.get("generator"),
        "Query rewriter": models.get("rewriter"),
        "Relevance checker": models.get("relevance_checker", shared_checker),
        "Groundedness checker": models.get("groundedness_checker", shared_checker),
        "Semantic judge": (
            deepeval_metrics.get("judge_model") if deepeval_metrics else models.get("judge")
        ),
    }


def case_issue_data(answers, qasper_scores, judge_scores, diagnostics, threshold):
    qasper_by_key = {case_key(item): item for item in (qasper_scores or [])}
    judge_by_key = {case_key(item): item for item in (judge_scores or [])}
    rows = []
    severity_order = {
        "generation error": 0,
        "judge error": 1,
        "failed semantic threshold": 2,
        "not grounded": 3,
        "partially grounded": 4,
        "unknown groundedness": 5,
        "low retrieval relevance": 6,
        "unknown relevance": 7,
    }
    for index, answer in enumerate(answers):
        key = case_key(answer)
        qasper = qasper_by_key.get(key)
        judge = judge_by_key.get(key)
        diagnostic = diagnostics[key]
        issues = []
        if answer.get("status") == "error":
            issues.append("generation error")
        if judge and judge.get("status") == "judge_error":
            issues.append("judge error")
        if judge and judge.get("status") in {"success", "generation_error"} and judge.get("passed") is False:
            issues.append("failed semantic threshold")
        groundedness = diagnostic["groundedness"]
        relevance = diagnostic["relevance"]
        if groundedness == "not_grounded":
            issues.append("not grounded")
        elif groundedness == "partially_grounded":
            issues.append("partially grounded")
        elif groundedness == "unknown":
            issues.append("unknown groundedness")
        if relevance == "low":
            issues.append("low retrieval relevance")
        elif relevance == "unknown":
            issues.append("unknown relevance")
        severity = min((severity_order[item] for item in issues), default=99)
        rows.append(
            {
                "index": index,
                "key": key,
                "answer": answer,
                "qasper": qasper,
                "judge": judge,
                "diagnostic": diagnostic,
                "issues": issues,
                "severity": severity,
            }
        )
    attention = sorted(
        (item for item in rows if item["issues"]),
        key=lambda item: (item["severity"], item["index"]),
    )
    return rows, attention


def render_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def validate_pipeline_artifacts(run_id, manifest, summary):
    if manifest:
        validate_run_id(manifest, "run_id", run_id, "Pipeline manifest")
    if summary:
        validate_run_id(summary, "run_id", run_id, "Pipeline summary")
        if manifest and summary.get("status") != manifest.get("status"):
            raise ReportValidationError("Pipeline status artifacts disagree.")
    if manifest and summary and (
        manifest.get("case_selection") != summary.get("case_selection")
    ):
        raise ReportValidationError(
            "Pipeline case-selection provenance artifacts disagree."
        )
    if manifest and manifest.get("pipeline_version") == CURRENT_PIPELINE_VERSION:
        expected_contract = semantic_evaluation_contract(V16_EVALUATOR_VERSION)
        if manifest.get("semantic_evaluation_contract") != expected_contract:
            raise ReportValidationError(
                "Pipeline semantic-evaluation provenance is invalid."
            )
        if summary and summary.get("semantic_evaluation_contract") != expected_contract:
            raise ReportValidationError(
                "Pipeline summary semantic-evaluation provenance is invalid."
            )
        report_stage = next(
            (
                stage for stage in manifest.get("stages", [])
                if isinstance(stage, dict) and stage.get("stage") == "report"
            ),
            None,
        )
        if report_stage is None:
            raise ReportValidationError("Pipeline report stage is missing.")
        manifest_status = report_stage.get("status")
        summary_report = (summary or {}).get("report") or {}
        summary_claims_complete = summary_report.get("status") == "complete"
        summary_complete = (
            summary_claims_complete
            and bool(summary_report.get("path"))
        )
        expected_report_path = (
            RUNS_DIR / run_id / DEFAULT_REPORT_NAME
        ).relative_to(REPO_ROOT).as_posix()

        summary_path_matches = (
            summary_report.get("path") == expected_report_path
        )
        if manifest_status == "complete" and not (
            summary_complete and summary_path_matches
        ):
            raise ReportValidationError(
                "Pipeline manifest and summary disagree about the report artifact."
            )
        if manifest_status in {"pending", "running"} and summary_claims_complete:
            raise ReportValidationError(
                "Pipeline summary completes a report stage that is still in progress."
            )
        if manifest_status == "skipped" and summary_claims_complete:
            raise ReportValidationError(
                "Pipeline summary completes a skipped report stage."
            )
        if manifest_status not in {"complete", "pending", "running", "skipped"}:
            if summary_claims_complete:
                raise ReportValidationError(
                    "Pipeline summary completes an unfinished report stage."
                )


def load_report_data(run_id, judge_id=None, skip_qasper=False, skip_deepeval=False):
    if not RUN_ID_RE.fullmatch(run_id):
        raise ReportValidationError("Invalid run ID.")
    run_dir = RUNS_DIR / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: benchmarking/runs/{run_id}")
    paths = {
        "generation_manifest": run_dir / "generation_manifest.json",
        "answers": run_dir / "answers.jsonl",
        "application_logs": run_dir / "application_logs.jsonl",
        "qasper_metrics": run_dir / "qasper_metrics.json",
        "qasper_scores": run_dir / "qasper_scores.jsonl",
        "pipeline_manifest": run_dir / "pipeline_manifest.json",
        "pipeline_summary": run_dir / "pipeline_summary.json",
    }
    pipeline_manifest = read_json(paths["pipeline_manifest"], "pipeline manifest", required=False)
    pipeline_summary = read_json(paths["pipeline_summary"], "pipeline summary", required=False)
    validate_pipeline_artifacts(run_id, pipeline_manifest, pipeline_summary)
    pipeline_judge_id = pipeline_recorded_judge_id(
        pipeline_manifest,
        pipeline_summary,
    )
    generation_manifest, answers, duplicate_generation_attempts, success_count, generation_errors = validate_generation(
        run_id, paths["generation_manifest"], paths["answers"]
    )
    for artifact, label in (
        (pipeline_manifest, "Pipeline manifest"),
        (pipeline_summary, "Pipeline summary"),
    ):
        if artifact and artifact.get("case_selection") != generation_manifest.get(
            "case_selection"
        ):
            raise ReportValidationError(
                f"{label} case-selection provenance contradicts generation."
            )
    application_logs = validate_application_logs(paths["application_logs"], answers)

    qasper_metrics = qasper_scores = None
    qasper_reason = None
    if skip_qasper or stage_status(pipeline_manifest, "qasper") == "skipped":
        qasper_reason = "The deterministic QASPER stage was explicitly skipped."
    elif paths["qasper_metrics"].is_file() or paths["qasper_scores"].is_file():
        if not (paths["qasper_metrics"].is_file() and paths["qasper_scores"].is_file()):
            raise ReportValidationError("QASPER artifacts are incomplete.")
        qasper_metrics, qasper_scores = validate_qasper(
            run_id, paths["qasper_metrics"], paths["qasper_scores"], answers,
            duplicate_generation_attempts,
        )
    else:
        qasper_reason = "Deterministic QASPER artifacts are not available."

    selected_judge_id = None
    judge_manifest = deepeval_metrics = judge_scores = judge_attempts = None
    deepeval_reason = None
    if skip_deepeval or stage_status(pipeline_manifest, "deepeval") == "skipped":
        deepeval_reason = "The semantic evaluation stage was explicitly skipped."
    else:
        selected_judge_id = select_judge_id(run_dir, judge_id, pipeline_manifest, pipeline_summary)
        if selected_judge_id:
            judge_dir = run_dir / "deepeval" / selected_judge_id
            paths.update({
                "judge_manifest": judge_dir / "judge_manifest.json",
                "deepeval_metrics": judge_dir / "metrics.json",
                "deepeval_scores": judge_dir / "scores.jsonl",
            })
            judge_manifest, deepeval_metrics, judge_scores, judge_attempts = validate_deepeval(
                run_id, selected_judge_id, paths["judge_manifest"],
                paths["deepeval_metrics"], paths["deepeval_scores"], answers,
                generation_manifest.get("case_selection"),
            )
        else:
            deepeval_reason = "DeepEval artifacts are not available."

    if pipeline_summary:
        generation_summary = pipeline_summary.get("generation") or {}
        if generation_summary.get("selected_questions") not in (None, len(answers)):
            raise ReportValidationError("Pipeline generation count is contradictory.")
        deterministic_summary = pipeline_summary.get("deterministic_qasper") or {}
        if qasper_metrics and deterministic_summary.get("answer_f1") not in (None, qasper_metrics["answer_f1"]):
            raise ReportValidationError("Pipeline QASPER metric is contradictory.")
    validate_matching_pipeline_judge(
        selected_judge_id,
        pipeline_judge_id,
        pipeline_manifest,
        pipeline_summary,
        deepeval_metrics,
    )

    alternate_judge_note = None
    if (
        judge_id is not None
        and selected_judge_id is not None
        and pipeline_judge_id is not None
        and selected_judge_id != pipeline_judge_id
    ):
        alternate_judge_note = (
            f"This report explicitly selects judge {selected_judge_id}, which "
            "differs from the judge recorded by the original pipeline run."
        )

    provenances = embedded_configuration(
        ("generation manifest", generation_manifest),
        ("QASPER metrics", qasper_metrics),
        ("judge manifest", judge_manifest),
    )
    configuration = benchmark_configuration(generation_manifest, judge_manifest, provenances)
    relevance_counts, groundedness_counts, diagnostics = diagnostic_labels(answers, application_logs)
    checker_accounting = generation_manifest.get("checker_diagnostics")
    if checker_accounting is not None:
        if not isinstance(checker_accounting, dict) or not application_logs:
            raise ReportValidationError(
                "Recorded checker diagnostic accounting is malformed."
            )
        calculated_checker_accounting = aggregate_checker_diagnostics(
            list(application_logs["by_case"].values())
        )
        if checker_accounting != calculated_checker_accounting:
            raise ReportValidationError(
                "Checker diagnostic accounting contradicts application logs."
            )
    latency, tokens = latency_and_usage(answers, application_logs, judge_scores)
    cases, attention = case_issue_data(
        answers, qasper_scores, judge_scores, diagnostics,
        deepeval_metrics.get("threshold") if deepeval_metrics else None,
    )
    return locals()


def semantic_values(data):
    metrics = data["deepeval_metrics"]
    total = len(data["answers"])
    if not metrics:
        return ("N/A", "N/A", "N/A", "N/A")
    non_null = metrics["non_null_evaluated_case_count"]
    selected = metrics["selected_case_count"]
    mean = metrics["mean_semantic_correctness"]
    passed = sum(item.get("passed") is True for item in data["judge_scores"])
    mean_text = (
        f"{mean!r} ({format_percentage(mean)}) across {non_null} non-null "
        "judged cases"
        if mean is not None
        else "N/A"
    )
    pass_text = (
        f"{metrics['pass_rate']!r} "
        f"({format_percentage(metrics['pass_rate'])}; {passed}/{non_null})"
        if metrics["pass_rate"] is not None
        else "N/A"
    )
    coverage = non_null / total if total else 0.0
    coverage_text = f"{format_percentage(coverage)} ({non_null}/{total})"
    full = metrics.get("full_run_semantic_correctness")
    if data["generation_manifest"].get("case_selection") is not None:
        full_text = "Not applicable (targeted failure-subset diagnostic)"
    else:
        full_text = (
            f"{full!r} ({format_percentage(full)})"
            if selected == total and coverage == 1.0 and full is not None
            else "N/A (semantic evaluation does not cover every selected case)"
        )
    return mean_text, pass_text, coverage_text, full_text


def metadata_rows(data):
    manifest = data["generation_manifest"]
    pipeline = data["pipeline_manifest"] or {}
    summary = data["pipeline_summary"] or {}
    config = data["configuration"]
    counts = manifest["counts"]
    return [
        ("Run ID", data["run_id"]),
        ("Overall status", summary.get("status", pipeline.get("status", manifest.get("status")))),
        ("Benchmark condition", (manifest.get("configuration") or {}).get("condition", "single_document").replace("_", " ")),
        ("Selected papers", counts.get("selected_papers")),
        ("Selected questions", counts.get("selected_questions")),
        ("Completion timestamp", summary.get("completion_timestamp", pipeline.get("completed_at", manifest.get("finished_at")))),
        ("Configuration profile", config["profile_name"] or "Not recorded (legacy run)"),
        ("Configuration digest", config["configuration_sha256"][:12] if config["configuration_sha256"] else "Not recorded"),
        ("Dataset digest", config["dataset_sha256"][:12] if config["dataset_sha256"] else "Not recorded"),
    ]


def render_report(data, report_path):
    answers = data["answers"]
    total = len(answers)
    qmetrics = data["qasper_metrics"]
    dmetrics = data["deepeval_metrics"]
    logs = data["application_logs"]
    relevance = data["relevance_counts"]
    groundedness = data["groundedness_counts"]
    semantic_mean, semantic_pass, semantic_coverage, full_semantic = semantic_values(data)
    answer_latency = series_summary(data["latency"]["End-to-end answer"])
    valid_relevance = total - relevance["unknown"] - relevance["missing"]
    valid_groundedness = total - groundedness["unknown"] - groundedness["missing"]
    case_selection = data["generation_manifest"].get("case_selection")

    lines = REPORT_HEADER.splitlines()
    lines.extend(f"- **{label}:** {markdown_text(value)}" for label, value in metadata_rows(data))
    if case_selection is not None:
        lines.extend(
            [
                "",
                "## Targeted Failure-Subset Diagnostic",
                "",
                "**This is a selection-biased failure-recovery analysis, not a new full-benchmark result.**",
                "",
                "Its semantic mean and pass rate describe only cases that scored strictly below the source judge threshold. They must be interpreted as failure-recovery performance and must not be combined with the previous full-run metric.",
                "",
                render_table(
                    ["Selection provenance", "Recorded value"],
                    [
                        ["Selection mode", case_selection["selection_mode"]],
                        ["Source run", markdown_text(case_selection["source_run_id"])],
                        ["Source judge", markdown_text(case_selection["source_judge_id"])],
                        ["Source evaluator", markdown_text(case_selection["source_evaluator_version"])],
                        ["Source threshold", case_selection["source_threshold"]],
                        ["Selected cases", case_selection["selected_case_count"]],
                        ["Ordered case-key SHA-256", markdown_text(case_selection["ordered_case_keys_sha256"])],
                        ["Source dataset SHA-256", markdown_text(case_selection["source_dataset_sha256"])],
                    ],
                ),
            ]
        )
    lines.extend(["", "## Executive Summary", ""])
    lines.append(render_table(["Measure", "Result"], [
        ["Generation success", format_ratio(data["success_count"], total)],
        ["QASPER Answer F1", format_percentage(qmetrics["answer_f1"]) if qmetrics else "Not available"],
        ["Semantic correctness among non-null judged cases", semantic_mean],
        ["Semantic pass rate", semantic_pass],
        ["Semantic evaluation coverage", semantic_coverage],
        ["Retrieval relevance valid-result coverage", format_ratio(valid_relevance, total) if logs else "Not available"],
        ["Groundedness valid-result coverage", format_ratio(valid_groundedness, total) if logs else "Not available"],
        ["End-to-end median latency", f"{answer_latency['median']:.3f} s across {answer_latency['count']} cases"],
    ]))
    lines.extend(["", f"Full-run semantic correctness: **{full_semantic}**."])
    if data.get("alternate_judge_note"):
        lines.append(markdown_text(data["alternate_judge_note"]))
    if case_selection is not None:
        lines.append(
            "Full-run semantic correctness is intentionally not reported for this targeted subset."
        )
    elif dmetrics and dmetrics.get("full_run_semantic_correctness") is None:
        lines.append("A null full-run score is appropriate because at least one selected case has no semantic judgment; judge errors are not converted to zero.")
    lines.extend(["", "No composite metric is calculated.", "", "## Evaluation Status and Coverage", ""])
    judge_counts = data["judge_manifest"].get("counts", {}) if data["judge_manifest"] else {}
    checker_accounting = data.get("checker_accounting")
    checker_unknown = (
        sum(item["unknown"] for item in checker_accounting.values())
        if checker_accounting
        else None
    )
    lines.append(render_table(["Stage or condition", "Status"], [
        ["Answer generation", f"{data['success_count']}/{total} successful; {data['generation_errors']} generation errors"],
        ["Diagnostic checker availability", (
            f"complete; 0 unknown" if checker_unknown == 0
            else f"incomplete; {checker_unknown} unknown" if checker_unknown is not None
            else "Not recorded"
        )],
        ["Deterministic QASPER evaluator", "complete" if qmetrics else "Not available"],
        ["Semantic judge", data["judge_manifest"].get("status") if data["judge_manifest"] else "Not available"],
        ["Semantic selected cases", dmetrics.get("selected_case_count") if dmetrics else "N/A"],
        ["Successful judgments", dmetrics.get("successful_judgment_count") if dmetrics else "N/A"],
        ["Generation errors represented in semantic evaluation", dmetrics.get("generation_error_count") if dmetrics else "N/A"],
        ["Judge errors", dmetrics.get("judge_error_count") if dmetrics else "N/A"],
        ["Missing semantic cases", total - dmetrics.get("selected_case_count", 0) if dmetrics else "N/A"],
        ["Duplicate generation attempts", data["duplicate_generation_attempts"]],
        ["Historical judge-error attempts", judge_counts.get("judge_error_attempt_count", "Not recorded")],
        ["Unresolved judge errors", judge_counts.get("unresolved_judge_error_count", dmetrics.get("judge_error_count") if dmetrics else "N/A")],
        ["Application-log coverage", format_ratio(logs["matched_case_count"], total) if logs else "Not available"],
    ]))
    if data["qasper_reason"]:
        lines.extend(["", f"Deterministic evaluation: **Not available.** {markdown_text(data['qasper_reason'])}"])
    if data["deepeval_reason"]:
        lines.extend(["", f"Semantic evaluation: **Not available.** {markdown_text(data['deepeval_reason'])}"])
    warnings = []
    if logs:
        if logs["missing_conversation_ids"]:
            warnings.append(f"{len(logs['missing_conversation_ids'])} selected conversations have no matching application log.")
        if logs["duplicate_conversation_ids"]:
            warnings.append(f"{len(logs['duplicate_conversation_ids'])} selected conversations have duplicate matching application logs; their diagnostics are treated as missing.")
    if relevance["unknown"]:
        warnings.append(f"{relevance['unknown']} relevance results are unknown and unavailable, not negative labels.")
    if groundedness["unknown"]:
        warnings.append(f"{groundedness['unknown']} groundedness results are unknown and unavailable, not negative labels.")
    if dmetrics and dmetrics["judge_error_count"]:
        warnings.append(f"{dmetrics['judge_error_count']} judge error prevents full semantic coverage.")
    lines.extend([f"\nWarning: {markdown_text(item)}" for item in warnings])
    diagnostic_incomplete = (
        relevance["unknown"]
        or relevance["missing"]
        or groundedness["unknown"]
        or groundedness["missing"]
    )
    primary_evaluation_incomplete = (
        case_selection is not None
        or data["success_count"] != total
        or qmetrics is None
        or dmetrics is None
        or dmetrics.get("selected_case_count") != total
        or dmetrics.get("evaluation_coverage") != 1.0
        or dmetrics.get("judge_error_count") != 0
        or logs is None
        or logs.get("matched_case_count") != total
        or bool(logs.get("duplicate_conversation_ids"))
    )
    if primary_evaluation_incomplete:
        readiness = (
            "Benchmark-run publication readiness: **Not ready because required generation or evaluation coverage is incomplete.**"
        )
    elif diagnostic_incomplete:
        readiness = (
            "Benchmark-run publication readiness: **Ready with a minor diagnostic-coverage warning.** All selected answers and primary evaluation results are complete; unavailable checker labels are reported separately."
        )
    else:
        readiness = "Benchmark-run publication readiness: **Ready.**"
    lines.extend(["", readiness])

    lines.extend(["", "## Benchmark Configuration", ""])
    config = data["configuration"]["resolved"]
    config_rows = [
        ("QA data", config.get("qa_data_path")), ("PDF directory", config.get("pdf_directory")),
        ("Retrieval mode", config.get("retrieval_mode")), ("Chunk size", config.get("chunk_size")),
        ("Chunk overlap", config.get("chunk_overlap")), ("Retrieval Top N", config.get("retrieval_top_n")),
        ("Fusion threshold", config.get("fusion_threshold")), ("Rerank candidates", config.get("rerank_candidate_count")),
        ("Rerank Top K", config.get("rerank_top_k")), ("Judge threshold", config.get("judge_threshold")),
        ("Judge reasoning effort", config.get("judge_reasoning_effort")), ("Judge retry limit", config.get("judge_max_retries")),
        ("Judge completion-token limit", config.get("judge_max_completion_tokens")),
        ("Judge request interval (seconds)", config.get("judge_request_interval_seconds")),
        ("Judge retry base delay (seconds)", config.get("judge_retry_base_delay_seconds")),
        ("Judge retry maximum delay (seconds)", config.get("judge_retry_max_delay_seconds")),
        ("Judge retry jitter bound (seconds)", config.get("judge_retry_jitter_seconds")),
        ("Judge maximum Retry-After (seconds)", config.get("judge_max_retry_after_seconds")),
        ("Generation case interval (seconds)", config.get("generation_case_interval_seconds")),
    ]
    lines.append(render_table(["Setting", "Recorded value"], [[markdown_text(a), markdown_text(display_value(b))] for a, b in config_rows]))

    lines.extend(["", "## Models", ""])
    models = extract_models(data["generation_manifest"], dmetrics)
    lines.append(render_table(["Role", "Model"], [[role, markdown_text(display_value(model))] for role, model in models.items()]))

    lines.extend(["", "## Answer Correctness", ""])
    if qmetrics:
        lines.extend([
            f"- QASPER Answer F1: **{qmetrics['answer_f1']:.12f}** ({qmetrics['answer_f1_percentage']:.8f}%).",
            f"- Selected/evaluated cases: **{qmetrics['unique_evaluated_case_count']}/{qmetrics['selected_question_count']}**.",
            f"- Generation errors: **{qmetrics['generation_error_count']}**; recorded generation errors score zero.",
            "- Evidence F1: **Not computed.** " + markdown_text(qmetrics.get("evidence_f1_note")),
            "", render_table(["Matched answer type", "Mean Answer F1"], [[kind, f"{qmetrics['answer_f1_by_matched_answer_type'][kind]:.6f}"] for kind in QASPER_TYPES]),
        ])
    else:
        lines.append(f"**Not available.** {markdown_text(data['qasper_reason'])}")
    lines.append("")
    if dmetrics:
        evaluator_version = dmetrics["evaluator_version"]
        if evaluator_version == V16_EVALUATOR_VERSION:
            reference_aggregation = (
                "maximum over independently judged annotations; "
                "annotation-specific gold evidence is supplied; QASPER "
                "cross-reference placeholders are treated as non-semantic "
                "annotation markup; absence from an abbreviated reference is "
                "not itself evidence of fabrication"
            )
        elif evaluator_version == V15_EVALUATOR_VERSION:
            reference_aggregation = (
                "maximum over independently judged annotations; QASPER "
                "cross-reference placeholders are treated as non-semantic "
                "annotation markup"
            )
        elif evaluator_version == V14_EVALUATOR_VERSION:
            reference_aggregation = (
                "maximum over independently judged annotations"
            )
        elif evaluator_version == "qasper-reference-geval-v1.3":
            reference_aggregation = (
                "Single judge prompt containing alternative references; "
                "maximum was not programmatically enforced."
            )
        else:
            reference_aggregation = "Not recorded for this legacy artifact."
        lines.extend([
            f"- Semantic mean: **{semantic_mean}**.",
            f"- Semantic pass rate: **{semantic_pass}** at threshold **{dmetrics['threshold']:.2f}**.",
            f"- Semantic coverage: **{semantic_coverage}**.",
            f"- Full-run semantic correctness: **{full_semantic}**.",
            f"- Judge: **{markdown_text(dmetrics['judge_model'])}**; evaluator version **{markdown_text(dmetrics['evaluator_version'])}**.",
            f"- Reference aggregation: **{markdown_text(reference_aggregation)}**",
            "", render_table(["Rubric band", "Count"], [[markdown_text(k), v] for k, v in dmetrics.get("score_counts_by_rubric_band", {}).items()]),
        ])
        if evaluator_version == V16_EVALUATOR_VERSION:
            lines.extend(
                [
                    "",
                    "- Annotation evidence: **use `highlighted_evidence` when usable; otherwise use `evidence`; otherwise state that no gold evidence was supplied. Original evidence order is preserved.**",
                    "- Abbreviated-reference policy: **absence of a detail from the short reference is not itself evidence of fabrication.**",
                ]
            )
        if evaluator_version in INDEPENDENT_REFERENCE_VERSIONS:
            lines.extend([
                "",
                render_table(
                    ["Reference-judgment measure", "Count"],
                    [
                        ["Total expected reference judgments", dmetrics["expected_reference_judgment_count"]],
                        ["Successful reference judgments", dmetrics["successful_reference_judgment_count"]],
                        ["Failed reference judgments", dmetrics["reference_judgment_error_count"]],
                        ["Cases where a non-first annotation won", dmetrics["non_first_winning_reference_count"]],
                    ],
                ),
                "",
                render_table(
                    ["Matched reference answer type", "Cases"],
                    [[kind, dmetrics["cases_matched_by_answer_type"][kind]] for kind in QASPER_TYPES],
                ),
            ])
    else:
        lines.append(f"Semantic evaluation: **Not available.** {markdown_text(data['deepeval_reason'])}")
    lines.extend(["", "### Judge Reliability", ""])
    reliability = (
        dmetrics.get("transport_reliability")
        if dmetrics
        and dmetrics.get("evaluator_version") in RELIABILITY_EVALUATOR_VERSIONS
        else None
    )
    if reliability:
        lines.append(render_table(["Reliability measure", "Recorded value"], [
            ["Total provider attempts", reliability["total_provider_attempts"]],
            ["Cases requiring at least one retry", reliability["cases_requiring_retry_count"]],
            ["Total retries", reliability["total_retry_count"]],
            ["Rate-limit events", reliability["total_rate_limit_response_count"]],
            ["Rate-limit retries", reliability["total_rate_limit_retry_count"]],
            ["Other transient retries", reliability["total_other_transient_retry_count"]],
            ["Schema/output retries", reliability["total_schema_output_retry_count"]],
            ["Pacing sleep", f"{reliability['total_pacing_sleep_seconds']:.3f} s"],
            ["Retry/backoff sleep", f"{reliability['total_retry_backoff_sleep_seconds']:.3f} s"],
            ["Telemetry-covered selected cases", reliability["telemetry_covered_selected_case_count"]],
            ["Unresolved judge errors", reliability["unresolved_judge_error_count"]],
            ["Historical judge-error attempts", reliability["historical_judge_error_attempt_count"]],
        ]))
    elif dmetrics:
        lines.append(
            "**Not recorded.** This legacy judge artifact predates v1.2 "
            "transport telemetry; zero values are not inferred."
        )
    else:
        lines.append("**Not available.** Semantic evaluation artifacts are absent or skipped.")
    lines.extend(["", "Token F1 measures normalized lexical overlap against short references. Semantic correctness measures reference-based answer correctness and permits valid paraphrasing. These metrics measure different properties and must not be combined; semantic evaluation does not supersede deterministic F1."])
    lines.extend(["", "## RAG Diagnostics", ""])
    if logs:
        lines.append(render_table(["Retrieval relevance", "Count", "Share"], [
            [label, relevance[label], format_percentage(relevance[label] / total)] for label in RELEVANCE_LABELS
        ]))
        lines.extend(["", render_table(["Groundedness", "Count", "Share"], [
            [label, groundedness[label], format_percentage(groundedness[label] / total)] for label in GROUNDEDNESS_LABELS
        ])])
        lines.extend(["", "Unknown means that a checker result was unavailable; it is not low relevance, not-grounded, or an incorrect answer. These checker labels are operational diagnostics, not manually validated accuracy metrics."])
        checker_accounting = data.get("checker_accounting")
        if checker_accounting:
            cause_rows = []
            for role in ("relevance", "groundedness"):
                values = checker_accounting[role]
                cause_rows.append([role.title(), "valid", values["valid"]])
                cause_rows.append([role.title(), "unknown", values["unknown"]])
                for cause, count in values["unknown_by_cause"].items():
                    cause_rows.append([role.title(), cause, count])
            lines.extend([
                "",
                render_table(
                    ["Checker", "Availability/cause", "Count"],
                    cause_rows,
                ),
                "",
                "TPM pacing reduces request bursts but does not increase or reset TPD allowance. TPD exhaustion cannot be repaired by semantic-judge resume; generation/checker diagnostics require sufficient daily quota, and concurrent use of the shared openai/gpt-oss-20b checker model can consume it.",
            ])
        else:
            lines.extend([
                "",
                "Operational causes: **Not recorded.** This historical generation manifest predates structured checker-failure accounting.",
            ])
    else:
        lines.append("**Not available.** Application-log diagnostics are absent.")

    lines.extend(["", "## Latency and Usage", ""])
    latency_rows = []
    for name, values in data["latency"].items():
        summary = series_summary(values)
        if summary:
            latency_rows.append([name, summary["count"], f"{summary['mean']:.3f}", f"{summary['median']:.3f}", f"{summary['p95']:.3f}", f"{summary['maximum']:.3f}"])
    lines.append(render_table(["Series (seconds)", "n", "Mean", "Median", "p95", "Maximum"], latency_rows))
    tokens = data["tokens"]
    token_usage_lines = recorded_token_usage_lines(tokens)
    lines.extend([
        "", *token_usage_lines,
        "", "p95 uses the deterministic nearest-rank definition. Checker calls run concurrently in the production chat path; checker latencies are not added together or presented as production wall-clock latency.",
        "", "## Cases Requiring Attention", "",
    ])
    attention_rows = []
    for item in data["attention"]:
        answer, qscore, judge, diagnostic = item["answer"], item["qasper"], item["judge"], item["diagnostic"]
        attention_rows.append([
            markdown_text(answer["paper_id"]), markdown_text(short_question_id(answer["question_id"])), markdown_text(answer["question"], 90),
            markdown_text(answer.get("status")), format_number(qscore.get("answer_f1") if qscore else None),
            format_number(judge.get("semantic_score") if judge else None),
            markdown_text(display_value(judge.get("matched_answer_type") if judge else None)),
            diagnostic["relevance"], diagnostic["groundedness"], markdown_text(", ".join(item["issues"])),
        ])
    lines.append(render_table(["Paper ID", "Question ID", "Question", "Generation", "QASPER F1", "Semantic", "Matched reference type", "Relevance", "Groundedness", "Issue"], attention_rows))
    lines.extend(["", "Low token F1 alone is not treated as factual incorrectness.", "", "## Per-Question Results", ""])
    per_rows = []
    for item in data["cases"]:
        answer, qscore, judge, diagnostic = item["answer"], item["qasper"], item["judge"], item["diagnostic"]
        per_rows.append([
            markdown_text(answer["paper_id"]), markdown_text(short_question_id(answer["question_id"])), markdown_text(answer["question"], 90), markdown_text(answer.get("status")),
            format_number(qscore.get("answer_f1") if qscore else None), format_number(judge.get("semantic_score") if judge else None),
            display_value(judge.get("passed") if judge else None), diagnostic["relevance"], diagnostic["groundedness"],
        ])
    lines.append(render_table(["Paper", "Question ID", "Question", "Generation", "QASPER F1", "Semantic", "Semantic pass", "Relevance", "Groundedness"], per_rows))

    lines.extend([
        "", "## Methodology and Limitations", "",
        "- This is a single-document evaluation whose questions are independent and begin with empty chat history.",
        "- QASPER Answer F1 takes the maximum normalized token F1 over valid human annotations.",
        "- Explanatory answers can receive low token F1 against short gold spans.",
        "- DeepEval is an LLM-as-a-judge measurement and remains nondeterministic.",
        (
            "- Semantic reference aggregation: " + reference_aggregation
            if dmetrics
            else "- Semantic reference aggregation is not available."
        ),
        "- Checker labels are diagnostic and were not manually calibrated for accuracy.",
        "- Evidence F1 is unavailable because retrieved chunks are not mapped to QASPER paragraph evidence.",
        "- The selected corpus is a custom subset, so results are not directly comparable with the official full-split QASPER baseline.",
        "- Results apply only to the recorded dataset, configuration, and model versions.",
        (
            "- This targeted subset is selection-biased and measures failure-recovery performance only; it is not a full-benchmark result."
            if case_selection is not None
            else "- No prior-score failure-subset selection was applied."
        ),
        "- Incomplete required generation or semantic-evaluation coverage blocks publication; incomplete auxiliary checker diagnostics are disclosed as warnings.",
        "", "## Reproducibility", "",
    ])
    config_meta = data["configuration"]
    pipeline = data["pipeline_manifest"] or {}
    versions = [
        ["Profile name", markdown_text(display_value(config_meta["profile_name"]))],
        ["Configuration SHA-256", markdown_text(display_value(config_meta["configuration_sha256"]))],
        ["Dataset SHA-256", markdown_text(display_value(config_meta["dataset_sha256"]))],
        ["Git commit", markdown_text(display_value(pipeline.get("git_commit")))],
        ["Git dirty", markdown_text(display_value(pipeline.get("git_dirty")))],
        ["Generation version", markdown_text(display_value(data["generation_manifest"].get("generator_version") or data["generation_manifest"].get("generation_version")))],
        ["QASPER evaluator version", markdown_text(display_value((qmetrics or {}).get("evaluator_version") if qmetrics else None))],
        ["Semantic evaluator version", markdown_text(display_value((dmetrics or {}).get("evaluator_version")))],
    ]
    lines.append(render_table(["Provenance", "Recorded value"], versions))
    lines.extend(["", "Source artifacts:", ""])
    for name in ("generation_manifest", "answers", "application_logs", "qasper_metrics", "qasper_scores", "pipeline_manifest", "pipeline_summary", "judge_manifest", "deepeval_metrics", "deepeval_scores"):
        if name in data["paths"]:
            lines.append("- " + artifact_link(report_path, data["paths"][name], name.replace("_", " ").title()))
    return "\n".join(lines).rstrip() + "\n" + REPORT_FOOTER


def validate_report_safety(content):
    forbidden = [
        r"C:\Users", "OneDrive", str(REPO_ROOT), str(REPO_ROOT).replace("\\", "/"),
        tempfile.gettempdir(), tempfile.gettempdir().replace("\\", "/"),
        "Bearer ", "authorization:", "gsk_",
    ]
    lowered = content.lower()
    for value in forbidden:
        if value and value.lower() in lowered:
            raise ReportValidationError("Report contains a private path or secret marker.")
    # The approved public identity can contain the local username as a substring.
    # Exempt only exact fixed branding at its intended boundaries from that check;
    # all path and secret checks above still cover the entire report.
    evaluation_content = content.removeprefix(REPORT_HEADER).removesuffix(REPORT_FOOTER)
    username = os.environ.get("USERNAME", "")
    if username and username.lower() in evaluation_content.lower():
        raise ReportValidationError("Report contains a private path or secret marker.")


def generate_report(run_id, output=None, judge_id=None, skip_qasper=False, skip_deepeval=False):
    report_path = resolve_output_path(output, RUNS_DIR / run_id)
    data = load_report_data(run_id, judge_id, skip_qasper, skip_deepeval)
    content = render_report(data, report_path)
    validate_report_safety(content)
    atomic_write_text(report_path, content)
    return report_path, data


def build_parser():
    parser = argparse.ArgumentParser(description="Generate a deterministic Markdown report from existing evaluation artifacts.")
    parser.add_argument("--run-id", required=True, help="Existing generation run ID.")
    parser.add_argument("--judge-id", help="Completed semantic judge run to report.")
    parser.add_argument("--skip-qasper", action="store_true", help="Exclude deterministic QASPER artifacts, including stale files.")
    parser.add_argument("--skip-deepeval", action="store_true", help="Exclude DeepEval artifacts, including stale directories.")
    parser.add_argument("--output", help="Repository-relative POSIX .md output path.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        path, _ = generate_report(args.run_id, args.output, args.judge_id, args.skip_qasper, args.skip_deepeval)
    except (FileNotFoundError, ReportValidationError, OSError) as exc:
        parser.exit(1, f"Report generation failed: {exc}\n")
    print(f"Report written: {portable_path(path)}")


if __name__ == "__main__":
    main()
