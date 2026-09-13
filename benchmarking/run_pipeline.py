"""Subprocess orchestrator for the QASPER evaluation pipeline and report."""

import argparse
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmarking.evaluation_config import (
    PATH_BASE,
    REPO_ROOT,
    LEGACY_RESOLVED_CONFIG_FILENAME,
    config_from_provenance,
    default_reasoning_effort,
    git_provenance as shared_git_provenance,
    load_evaluation_config,
    load_legacy_evaluation_config,
    portable_command,
    profile_provenance,
    sanitize_error as sanitize_error_text,
    sha256_json,
    serialize_artifact_path,
    validate_generation_profile_scope,
    validate_generation_profile_provenance,
    validate_profile_provenance,
)
from benchmarking.evaluate_qasper_f1 import load_manifest as load_generation_manifest
from benchmarking.evaluate_qasper_f1 import read_answer_attempts, validate_gold_bundle
from benchmarking.artifact_io import atomic_write_json
from benchmarking.checker_diagnostics import aggregate_checker_diagnostics
from benchmarking.case_selection import CaseSelectionError, load_below_threshold_selection
from benchmarking.qasper_judge_input import (
    ANNOTATION_EVIDENCE_POLICY,
    evidence_for_reference,
    semantic_evaluation_contract,
)


EVAL_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVAL_DIR / "runs"
PIPELINE_VERSION = "qasper-eval-pipeline-v1.6"
DEEPEVAL_VERSION = "qasper-reference-geval-v1.6"
SEMANTIC_EVALUATION_CONTRACT = semantic_evaluation_contract(DEEPEVAL_VERSION)
ACCEPTED_JUDGE_STATUSES = {"complete", "complete_with_judge_errors"}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SENSITIVE_OPTIONS = {
    "--api-key",
    "--authorization",
    "--authorization-header",
    "--token",
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


class PipelineError(RuntimeError):
    """A stage or artifact failed pipeline validation."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be at least 0")
    return parsed


def threshold_value(value):
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return parsed


def safe_judge_id(model_id):
    judge_id = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-._")
    if not judge_id:
        raise ValueError("Could not derive a safe judge ID from the model ID.")
    return judge_id.lower()


def redact_text(value):
    return sanitize_error_text(value)


def sanitize_arguments(arguments):
    """Return command arguments safe for persistence and terminal display."""
    sanitized = []
    redact_next = False
    for argument in map(str, arguments):
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        option, separator, _ = argument.partition("=")
        if option.lower() in SENSITIVE_OPTIONS:
            if separator:
                sanitized.append(f"{option}=<redacted>")
            else:
                sanitized.append(option)
                redact_next = True
            continue
        sanitized.append(redact_text(argument))
    return sanitized


def concise_error(error):
    message = " ".join(str(error).split()) or type(error).__name__
    return sanitize_error_text(message)[:1000]


def load_json_object(path, label):
    if not path.is_file():
        raise PipelineError(f"{label} not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"invalid numeric constant {value}")
                ),
            )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise PipelineError(f"Malformed {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"{label} must contain a JSON object: {path}")
    return payload


def _stage_mapping(manifest):
    stages = manifest.get("stages")
    if not isinstance(stages, list) or len(stages) != 4:
        raise PipelineError("Pipeline manifest has an invalid stage list.")
    mapping = {}
    for stage in stages:
        if not isinstance(stage, dict) or stage.get("stage") not in {
            "generation", "qasper", "deepeval", "report",
        }:
            raise PipelineError("Pipeline manifest has an invalid stage record.")
        if stage["stage"] in mapping:
            raise PipelineError("Pipeline manifest contains duplicate stages.")
        mapping[stage["stage"]] = stage
    if set(mapping) != {"generation", "qasper", "deepeval", "report"}:
        raise PipelineError("Pipeline manifest is missing a required stage.")
    return mapping


def _validate_recovery_provenance(
    candidate,
    run_id,
    generation_profile,
    effective_judge_profile,
    case_selection,
):
    if candidate.get("pipeline_version") != PIPELINE_VERSION:
        raise PipelineError("Stale temporary manifest has an incompatible version.")
    if candidate.get("semantic_evaluation_contract") != SEMANTIC_EVALUATION_CONTRACT:
        raise PipelineError(
            "Stale temporary manifest has incompatible semantic-evaluation provenance."
        )
    if candidate.get("run_id") != run_id:
        raise PipelineError("Stale temporary manifest belongs to another run.")
    for field, profile in (
        ("generation_profile", generation_profile),
        ("effective_judge_profile", effective_judge_profile),
    ):
        provenance = candidate.get(field)
        if not isinstance(provenance, dict) or (
            provenance.get("configuration_sha256")
            != profile.configuration_sha256
            or provenance.get("dataset_sha256") != profile.dataset_sha256
        ):
            raise PipelineError(
                f"Stale temporary manifest has incompatible {field} provenance."
            )
    if candidate.get("case_selection") != case_selection:
        raise PipelineError(
            "Stale temporary manifest has incompatible case-selection provenance."
        )


def _candidate_advances_target(candidate, target):
    candidate_stages = _stage_mapping(candidate)
    if target is None:
        return True
    target_stages = _stage_mapping(target)
    completed = {"complete", "complete_with_errors", "complete_with_judge_errors", "skipped"}
    advanced = False
    for name in candidate_stages:
        old_status = target_stages[name].get("status")
        new_status = candidate_stages[name].get("status")
        if old_status in completed and new_status not in completed:
            raise PipelineError(
                f"Stale temporary manifest would downgrade completed stage {name}."
            )
        if old_status not in completed and new_status in completed:
            advanced = True
        elif old_status in completed and new_status != old_status:
            raise PipelineError(
                f"Stale temporary manifest contradicts completed stage {name}."
            )
    if not advanced:
        raise PipelineError(
            "Stale temporary manifest is not demonstrably more advanced."
        )
    return True


def validate_pipeline_report_agreement(run_dir, manifest, summary):
    report_stage = _stage_mapping(manifest)["report"]
    summary_report = summary.get("report") if isinstance(summary, dict) else None
    report_exists = (run_dir / "evaluation_report.md").is_file()
    manifest_complete = report_stage.get("status") == "complete"
    summary_complete = (
        isinstance(summary_report, dict)
        and summary_report.get("status") == "complete"
        and bool(summary_report.get("path"))
    )
    if not (manifest_complete == summary_complete == report_exists):
        raise PipelineError(
            "Pipeline manifest, summary, and report artifact disagree."
        )


def _validated_state_from_manifest(
    run_dir,
    manifest,
    generation_profile,
    effective_judge_profile,
    case_selection,
):
    stages = _stage_mapping(manifest)
    generation = validate_generation_artifacts(
        run_dir,
        manifest["run_id"],
        generation_profile,
        True,
        case_selection,
    )
    qasper = None
    if stages["qasper"].get("status") != "skipped":
        if stages["qasper"].get("status") != "complete":
            raise PipelineError("Recovered QASPER stage is not complete.")
        qasper = validate_qasper_artifacts(
            run_dir,
            manifest["run_id"],
            generation,
            generation_profile,
        )
    deepeval = None
    if stages["deepeval"].get("status") != "skipped":
        judge_id = (manifest.get("invocation_configuration") or {}).get("judge_id")
        judge_model = effective_judge_profile.judge_model
        max_questions = (manifest.get("invocation_configuration") or {}).get(
            "judge_max_questions"
        )
        deepeval = validate_deepeval_artifacts(
            run_dir,
            manifest["run_id"],
            judge_id,
            judge_model,
            generation,
            max_questions,
            effective_judge_profile,
        )
    if stages["report"].get("status") != "complete":
        raise PipelineError("Recovered report stage is not complete.")
    report = validate_report_artifact(run_dir, manifest["run_id"])
    return generation, qasper, deepeval, report


def recover_stale_pipeline_manifest(
    run_dir,
    run_id,
    generation_profile,
    effective_judge_profile,
    case_selection,
):
    """Validate and recover only an unambiguously advanced complete manifest."""
    target_path = run_dir / "pipeline_manifest.json"
    temporary_path = target_path.with_suffix(target_path.suffix + ".tmp")
    if not temporary_path.exists():
        return None
    candidate = load_json_object(temporary_path, "temporary pipeline manifest")
    target = (
        load_json_object(target_path, "pipeline manifest")
        if target_path.is_file()
        else None
    )
    _validate_recovery_provenance(
        candidate,
        run_id,
        generation_profile,
        effective_judge_profile,
        case_selection,
    )
    if target is not None:
        _validate_recovery_provenance(
            target,
            run_id,
            generation_profile,
            effective_judge_profile,
            case_selection,
        )
    _candidate_advances_target(candidate, target)
    if candidate.get("status") not in {"complete", "complete_with_errors"}:
        raise PipelineError("Temporary pipeline manifest is not a completed run.")
    generation, qasper, deepeval, report = _validated_state_from_manifest(
        run_dir,
        candidate,
        generation_profile,
        effective_judge_profile,
        case_selection,
    )

    from benchmarking.artifact_io import atomic_write_text
    atomic_write_text(
        target_path,
        json.dumps(candidate, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    summary = build_summary(
        candidate,
        generation=generation,
        qasper=qasper,
        deepeval=deepeval,
        report=report,
    )
    atomic_write_json(run_dir / "pipeline_summary.json", summary)
    validate_pipeline_report_agreement(run_dir, candidate, summary)
    return summary


def load_jsonl(path, label):
    if not path.is_file():
        raise PipelineError(f"{label} not found: {path}")
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
        raise PipelineError(
            f"Malformed {label} at or before line {line_number}: {path}"
        ) from exc
    if not records:
        raise PipelineError(f"{label} contains no records: {path}")
    return records


def finite_number(value, label, minimum=None, maximum=None):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise PipelineError(f"{label} must be a finite number.")
    if minimum is not None and value < minimum:
        raise PipelineError(f"{label} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise PipelineError(f"{label} must be at most {maximum}.")
    return value


def validate_transport_telemetry(record, label):
    telemetry = record.get("judge_transport")
    if not isinstance(telemetry, dict):
        raise PipelineError(f"{label} lacks judge transport telemetry.")
    count_fields = (
        "provider_attempt_count",
        "retry_count",
        "rate_limit_response_count",
        "rate_limit_retry_count",
        "other_transient_retry_count",
        "schema_output_retry_count",
    )
    for field in count_fields:
        value = telemetry.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise PipelineError(f"{label} has invalid {field}.")
    for field in ("pacing_sleep_seconds", "retry_backoff_sleep_seconds"):
        finite_number(telemetry.get(field), f"{label} {field}", 0.0)
    if not isinstance(telemetry.get("retry_after_used"), bool):
        raise PipelineError(f"{label} has invalid retry_after_used.")
    if (
        telemetry["rate_limit_retry_count"]
        + telemetry["other_transient_retry_count"]
        + telemetry["schema_output_retry_count"]
        != telemetry["retry_count"]
    ):
        raise PipelineError(f"{label} retry categories do not reconcile.")
    if telemetry["retry_count"] > telemetry["provider_attempt_count"]:
        raise PipelineError(f"{label} retries exceed provider attempts.")
    if telemetry["rate_limit_retry_count"] > telemetry["rate_limit_response_count"]:
        raise PipelineError(f"{label} rate-limit retries exceed responses.")
    if telemetry["retry_after_used"] and telemetry["rate_limit_retry_count"] == 0:
        raise PipelineError(f"{label} used retry-after without a rate-limit retry.")
    category = telemetry.get("final_provider_error_category")
    if category is not None and (
        not isinstance(category, str) or not category.strip()
    ):
        raise PipelineError(f"{label} has invalid final provider error category.")
    if record.get("status") == "generation_error" and (
        any(telemetry[field] != 0 for field in count_fields)
        or telemetry["pacing_sleep_seconds"] != 0
        or telemetry["retry_backoff_sleep_seconds"] != 0
        or telemetry["retry_after_used"]
        or category is not None
    ):
        raise PipelineError(f"{label} generation error has nonzero judge telemetry.")
    return telemetry


def selected_deepeval_attempts(scores):
    grouped = {}
    for record in scores:
        grouped.setdefault(record.get("resume_key"), []).append(record)
    selected = []
    for records in grouped.values():
        successes = [item for item in records if item.get("status") == "success"]
        generation_errors = [
            item for item in records if item.get("status") == "generation_error"
        ]
        selected.append(
            successes[-1]
            if successes
            else generation_errors[-1]
            if generation_errors
            else records[-1]
        )
    return selected


def calculate_transport_reliability(scores):
    selected = selected_deepeval_attempts(scores)
    telemetry = [
        validate_transport_telemetry(record, "DeepEval selected score")
        for record in selected
    ]
    return selected, {
        "total_provider_attempts": sum(item["provider_attempt_count"] for item in telemetry),
        "cases_requiring_retry_count": sum(item["retry_count"] > 0 for item in telemetry),
        "total_retry_count": sum(item["retry_count"] for item in telemetry),
        "total_rate_limit_response_count": sum(item["rate_limit_response_count"] for item in telemetry),
        "total_rate_limit_retry_count": sum(item["rate_limit_retry_count"] for item in telemetry),
        "total_other_transient_retry_count": sum(item["other_transient_retry_count"] for item in telemetry),
        "total_schema_output_retry_count": sum(item["schema_output_retry_count"] for item in telemetry),
        "total_pacing_sleep_seconds": sum(item["pacing_sleep_seconds"] for item in telemetry),
        "total_retry_backoff_sleep_seconds": sum(item["retry_backoff_sleep_seconds"] for item in telemetry),
        "telemetry_covered_selected_case_count": len(telemetry),
        "unresolved_judge_error_count": sum(item.get("status") == "judge_error" for item in selected),
        "historical_judge_error_attempt_count": sum(item.get("status") == "judge_error" for item in scores),
    }


def validate_generation_artifacts(
    run_dir,
    run_id,
    profile=None,
    require_profile=False,
    expected_selection=None,
):
    manifest_path = run_dir / "generation_manifest.json"
    answers_path = run_dir / "answers.jsonl"
    try:
        manifest, selected_count = load_generation_manifest(
            manifest_path, run_id
        )
        selected_records, duplicate_count = read_answer_attempts(answers_path)
    except (OSError, ValueError, TypeError) as exc:
        raise PipelineError(
            f"Invalid generation artifacts for run {run_id!r}: {exc}"
        ) from exc
    if len(selected_records) != selected_count:
        raise PipelineError(
            "Generation selected-question count does not match unique answer "
            f"cases: {selected_count} != {len(selected_records)}."
        )
    if manifest.get("case_selection") != expected_selection:
        raise PipelineError(
            "Generation case-selection provenance does not match the pipeline."
        )
    if expected_selection is not None:
        ordered_keys = [
            f"{record['paper_id']}/{record['question_id']}"
            for record in selected_records
        ]
        if (
            expected_selection.get("selection_mode")
            != "prior_semantic_below_threshold"
            or expected_selection.get("selected_case_count") != selected_count
            or expected_selection.get("ordered_case_keys") != ordered_keys
            or expected_selection.get("ordered_case_keys_sha256")
            != sha256_json(ordered_keys)
            or expected_selection.get("source_dataset_sha256")
            != profile.dataset_sha256
        ):
            raise PipelineError(
                "Generation answers contradict failure-subset selection provenance."
            )
    if profile is not None:
        has_provenance = validate_generation_profile_provenance(
            manifest,
            profile,
            allow_legacy_read=not require_profile,
            allow_legacy_missing_case_pacing=True,
        )
        if require_profile and not has_provenance:
            raise PipelineError(
                "Generation artifacts do not contain evaluation profile provenance."
            )
    successful = sum(record["status"] == "success" for record in selected_records)
    checker_diagnostics = manifest.get("checker_diagnostics")
    recorded_configuration = (
        (manifest.get("evaluation_profile") or {}).get("resolved_configuration")
        or {}
    )
    if "generation_case_interval_seconds" in recorded_configuration:
        if not isinstance(checker_diagnostics, dict):
            raise PipelineError(
                "Generation manifest is missing checker diagnostic accounting."
            )
        application_logs_path = run_dir / "application_logs.jsonl"
        application_logs = []
        if application_logs_path.is_file():
            try:
                with application_logs_path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if line.strip():
                            record = json.loads(line)
                            if not isinstance(record, dict):
                                raise ValueError("record is not an object")
                            application_logs.append(record)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise PipelineError(
                    "Generation application logs are malformed."
                ) from exc
        calculated = aggregate_checker_diagnostics(application_logs)
        if checker_diagnostics != calculated:
            raise PipelineError(
                "Generation checker diagnostics contradict application logs."
            )
    return {
        "manifest": manifest,
        "selected_count": selected_count,
        "successful_count": successful,
        "generation_error_count": selected_count - successful,
        "duplicate_attempt_count": duplicate_count,
        "selected_records": selected_records,
        "artifacts": [
            serialize_artifact_path(manifest_path),
            serialize_artifact_path(answers_path),
        ],
    }


def validate_qasper_artifacts(run_dir, run_id, generation, profile=None):
    metrics_path = run_dir / "qasper_metrics.json"
    scores_path = run_dir / "qasper_scores.jsonl"
    metrics = load_json_object(metrics_path, "QASPER metrics")
    scores = load_jsonl(scores_path, "QASPER scores")
    selected_count = generation["selected_count"]

    if metrics.get("run_id") != run_id:
        raise PipelineError("QASPER metrics run ID does not match the pipeline run.")
    if metrics.get("selected_question_count") != selected_count:
        raise PipelineError(
            "QASPER selected-question count does not match generation."
        )
    if metrics.get("unique_evaluated_case_count") != selected_count:
        raise PipelineError(
            "QASPER unique evaluated count does not match generation selection."
        )
    finite_number(metrics.get("answer_f1"), "QASPER Answer F1", 0.0, 1.0)
    if metrics.get("evidence_f1", "missing") is not None:
        raise PipelineError("QASPER Evidence F1 must remain null in this stage.")
    if profile is not None and metrics.get("evaluation_profile") is not None:
        validate_generation_profile_provenance(metrics, profile)

    score_keys = []
    for index, record in enumerate(scores, start=1):
        paper_id = record.get("paper_id")
        question_id = record.get("question_id")
        if not isinstance(paper_id, str) or not isinstance(question_id, str):
            raise PipelineError(
                f"QASPER score record {index} has invalid case identifiers."
            )
        finite_number(
            record.get("answer_f1"),
            f"QASPER score record {index} Answer F1",
            0.0,
            1.0,
        )
        score_keys.append((paper_id, question_id))
    if len(score_keys) != selected_count or len(set(score_keys)) != selected_count:
        raise PipelineError(
            "QASPER scores do not contain exactly one record per selected case."
        )
    return {
        "metrics": metrics,
        "artifacts": [
            serialize_artifact_path(metrics_path),
            serialize_artifact_path(scores_path),
        ],
    }


def validate_v14_reference_record(score, answer, threshold, label):
    """Validate independent annotation judgments and deterministic max selection."""
    expected_references = validate_gold_bundle(
        answer["gold_answers"],
        f"{answer['paper_id']}/{answer['question_id']}",
    )
    references = score.get("references")
    if references != expected_references:
        raise PipelineError(f"{label} converted references contradict generation.")
    evaluations = score.get("reference_evaluations")
    if not isinstance(evaluations, list):
        raise PipelineError(f"{label} reference evaluations are missing.")
    if score["status"] == "generation_error":
        if evaluations or answer.get("status") != "error":
            raise PipelineError(
                f"{label} generation error must have no reference judgments."
            )
        return len(references), 0, 0, None, False
    if answer.get("status") != "success" or len(evaluations) != len(references):
        raise PipelineError(
            f"{label} must have one reference judgment per valid annotation."
        )

    successes = []
    errors = []
    aggregate_counts = {
        field: 0
        for field in (
            "provider_attempt_count",
            "retry_count",
            "rate_limit_response_count",
            "rate_limit_retry_count",
            "other_transient_retry_count",
            "schema_output_retry_count",
        )
    }
    aggregate_pacing = aggregate_backoff = 0.0
    retry_after_used = False
    for index, (reference, evaluation) in enumerate(zip(references, evaluations)):
        expected_identity = {
            "reference_index": index,
            "annotation_id": reference["annotation_id"],
            "answer_type": reference["answer_type"],
            "reference_answer": reference["answer"],
        }
        if any(evaluation.get(field) != value for field, value in expected_identity.items()):
            raise PipelineError(f"{label} reference judgment {index} is misaligned.")
        expected_evidence = evidence_for_reference(answer["gold_answers"], reference)
        if evaluation.get("judge_evidence") != expected_evidence:
            raise PipelineError(
                f"{label} reference judgment {index} uses evidence from the "
                "wrong annotation or source."
            )
        if evaluation.get("status") not in {"success", "judge_error"}:
            raise PipelineError(f"{label} reference judgment {index} has invalid status.")
        telemetry = validate_transport_telemetry(
            evaluation,
            f"{label} reference judgment {index}",
        )
        for field in aggregate_counts:
            aggregate_counts[field] += telemetry[field]
        aggregate_pacing += telemetry["pacing_sleep_seconds"]
        aggregate_backoff += telemetry["retry_backoff_sleep_seconds"]
        retry_after_used = retry_after_used or telemetry["retry_after_used"]
        if evaluation["status"] == "success":
            value = finite_number(
                evaluation.get("semantic_score"),
                f"{label} reference judgment {index} score",
                0.0,
                1.0,
            )
            if evaluation.get("passed") is not (value >= threshold):
                raise PipelineError(f"{label} reference pass value is contradictory.")
            if not isinstance(evaluation.get("judge_reason"), str) or evaluation.get("judge_error") is not None:
                raise PipelineError(f"{label} successful reference judgment is malformed.")
            successes.append(evaluation)
        else:
            if (
                evaluation.get("semantic_score") is not None
                or evaluation.get("passed") is not None
                or evaluation.get("judge_reason") is not None
                or not isinstance(evaluation.get("judge_error"), dict)
            ):
                raise PipelineError(f"{label} failed reference judgment is malformed.")
            errors.append(evaluation)

    case_transport = validate_transport_telemetry(score, label)
    if any(case_transport[field] != value for field, value in aggregate_counts.items()):
        raise PipelineError(f"{label} provider-attempt telemetry does not reconcile.")
    if (
        not math.isclose(case_transport["pacing_sleep_seconds"], aggregate_pacing, rel_tol=0.0, abs_tol=1e-6)
        or not math.isclose(case_transport["retry_backoff_sleep_seconds"], aggregate_backoff, rel_tol=0.0, abs_tol=1e-6)
        or case_transport["retry_after_used"] != retry_after_used
    ):
        raise PipelineError(f"{label} reference sleep telemetry does not reconcile.")

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
            raise PipelineError(f"{label} incomplete reference coverage is invalid.")
        return len(references), len(successes), len(errors), None, False

    winner = max(successes, key=lambda item: item["semantic_score"])
    expected_winner = {
        "matched_reference_index": winner["reference_index"],
        "matched_annotation_id": winner["annotation_id"],
        "matched_reference": winner["reference_answer"],
        "matched_answer_type": winner["answer_type"],
    }
    if score["status"] != "success" or any(
        score.get(field) != value for field, value in expected_winner.items()
    ):
        raise PipelineError(f"{label} maximum-reference selection is invalid.")
    if (
        score.get("semantic_score") != winner["semantic_score"]
        or score.get("passed") is not (winner["semantic_score"] >= threshold)
        or score.get("judge_reason") != winner["judge_reason"]
    ):
        raise PipelineError(f"{label} case result does not match its winning reference.")
    return (
        len(references),
        len(successes),
        0,
        winner["answer_type"],
        winner["reference_index"] != 0,
    )


def validate_deepeval_artifacts(
    run_dir,
    run_id,
    judge_id,
    judge_model,
    generation,
    judge_max_questions,
    profile=None,
):
    judge_dir = run_dir / "deepeval" / judge_id
    manifest_path = judge_dir / "judge_manifest.json"
    metrics_path = judge_dir / "metrics.json"
    scores_path = judge_dir / "scores.jsonl"
    manifest = load_json_object(manifest_path, "DeepEval judge manifest")
    metrics = load_json_object(metrics_path, "DeepEval metrics")
    scores = load_jsonl(scores_path, "DeepEval scores")

    expected_selected = generation["selected_count"]
    if judge_max_questions is not None:
        expected_selected = min(expected_selected, judge_max_questions)
    expected_identity = {
        "generation_run_id": run_id,
        "judge_id": judge_id,
        "judge_model": judge_model,
        "evaluator_version": DEEPEVAL_VERSION,
    }
    for field, expected in expected_identity.items():
        if manifest.get(field) != expected or metrics.get(field) != expected:
            raise PipelineError(
                f"DeepEval {field} does not match expected value {expected!r}."
            )
    status = manifest.get("status")
    if status not in ACCEPTED_JUDGE_STATUSES:
        raise PipelineError(f"DeepEval judge status {status!r} is not complete.")
    if profile is not None and manifest.get("evaluation_profile") is not None:
        validate_profile_provenance(manifest, profile)
        validate_profile_provenance(metrics, profile)
    case_selection = generation["manifest"].get("case_selection")
    if (
        manifest.get("case_selection") != case_selection
        or metrics.get("case_selection") != case_selection
    ):
        raise PipelineError(
            "DeepEval case-selection provenance contradicts generation."
        )
    placeholder_policy = manifest.get("qasper_placeholder_policy")
    if (
        not isinstance(placeholder_policy, dict)
        or metrics.get("qasper_placeholder_policy") != placeholder_policy
        or placeholder_policy.get("policy_version")
        != "qasper-placeholder-markup-v1"
        or placeholder_policy.get("reference_transformation") != "none"
        or placeholder_policy.get("original_converted_reference_preserved")
        is not True
    ):
        raise PipelineError("DeepEval QASPER placeholder provenance is invalid.")
    evidence_policy = manifest.get("qasper_annotation_evidence_policy")
    if (
        evidence_policy != ANNOTATION_EVIDENCE_POLICY
        or metrics.get("qasper_annotation_evidence_policy") != evidence_policy
    ):
        raise PipelineError(
            "DeepEval annotation-evidence provenance is invalid."
        )
    if (
        not isinstance(manifest.get("evaluation_steps"), list)
        or manifest.get("evaluation_steps") != metrics.get("evaluation_steps")
        or not isinstance(manifest.get("rubric"), list)
        or manifest.get("rubric") != metrics.get("rubric")
        or manifest.get("reference_semantics_note")
        != metrics.get("reference_semantics_note")
        or "absence from an abbreviated reference is not itself evidence of fabrication"
        not in str(manifest.get("reference_semantics_note", "")).lower()
    ):
        raise PipelineError("DeepEval judge methodology provenance is invalid.")
    manifest_selected = manifest.get("counts", {}).get("selected_cases")
    metrics_selected = metrics.get("selected_case_count")
    if manifest_selected != expected_selected or metrics_selected != expected_selected:
        raise PipelineError(
            "DeepEval selected-case count does not match the configured selection."
        )
    coverage = finite_number(
        metrics.get("evaluation_coverage"),
        "DeepEval evaluation coverage",
        0.0,
        1.0,
    )
    judge_errors = metrics.get("judge_error_count")
    successful_judgments = metrics.get("successful_judgment_count")
    generation_errors = metrics.get("generation_error_count")
    non_null_cases = metrics.get("non_null_evaluated_case_count")
    for value, label in (
        (judge_errors, "DeepEval judge error count"),
        (successful_judgments, "DeepEval successful judgment count"),
        (generation_errors, "DeepEval generation error count"),
        (non_null_cases, "DeepEval non-null evaluated case count"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise PipelineError(f"{label} must be a nonnegative integer.")
    if status == "complete" and judge_errors != 0:
        raise PipelineError("A complete DeepEval run cannot report judge errors.")
    if status == "complete_with_judge_errors" and judge_errors == 0:
        raise PipelineError(
            "DeepEval complete_with_judge_errors must report judge errors."
        )
    if non_null_cases != successful_judgments + generation_errors:
        raise PipelineError(
            "DeepEval non-null evaluated cases must equal successful judgments "
            "plus generation errors."
        )
    accounted_cases = successful_judgments + generation_errors + judge_errors
    if accounted_cases != expected_selected:
        raise PipelineError(
            "DeepEval success and error counts do not cover the selected cases."
        )
    expected_coverage = non_null_cases / expected_selected
    if not math.isclose(
        coverage,
        expected_coverage,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PipelineError(
            "DeepEval coverage does not match non-null evaluated cases divided "
            "by selected cases."
        )
    mean_score = metrics.get("mean_semantic_correctness")
    pass_rate = metrics.get("pass_rate")
    if non_null_cases == 0:
        if mean_score is not None or pass_rate is not None:
            raise PipelineError(
                "DeepEval mean and pass rate must be null when no semantic "
                "scores are available."
            )
    else:
        finite_number(
            mean_score,
            "DeepEval mean semantic correctness",
            0.0,
            1.0,
        )
        finite_number(pass_rate, "DeepEval pass rate", 0.0, 1.0)

    full_run_score = metrics.get("full_run_semantic_correctness")
    if math.isclose(coverage, 1.0, rel_tol=0.0, abs_tol=1e-12):
        if case_selection is not None:
            if full_run_score is not None:
                raise PipelineError(
                    "A targeted failure subset cannot report full-run semantic "
                    "correctness."
                )
        else:
            finite_number(
                full_run_score,
                "DeepEval full-run semantic correctness",
                0.0,
                1.0,
            )
    elif full_run_score is not None:
        raise PipelineError(
            "Partial DeepEval coverage cannot report full-run semantic correctness."
        )
    if len(scores) < expected_selected:
        raise PipelineError("DeepEval scores omit one or more selected cases.")
    for index, score in enumerate(scores, start=1):
        expected_resume_key = json.dumps(
            [
                score.get("paper_id"),
                score.get("question_id"),
                judge_model,
                DEEPEVAL_VERSION,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if (
            not isinstance(score.get("paper_id"), str)
            or not isinstance(score.get("question_id"), str)
            or score.get("judge_model") != judge_model
            or score.get("evaluator_version") != DEEPEVAL_VERSION
            or score.get("status")
            not in {"success", "generation_error", "judge_error"}
            or not isinstance(score.get("resume_key"), str)
            or score.get("resume_key") != expected_resume_key
        ):
            raise PipelineError(
                f"DeepEval score record {index} has incompatible identity or status."
            )
    selected_attempts, reliability = calculate_transport_reliability(scores)
    if len(selected_attempts) != expected_selected:
        raise PipelineError(
            "DeepEval append-only attempts do not resolve to one selected "
            "record per case."
        )
    selected_by_case = {
        (record["paper_id"], record["question_id"]): record
        for record in selected_attempts
    }
    generation_records = generation["selected_records"][:expected_selected]
    if len(selected_by_case) != expected_selected or set(selected_by_case) != {
        (record["paper_id"], record["question_id"])
        for record in generation_records
    }:
        raise PipelineError("DeepEval selected cases do not match generation cases.")
    reference_distribution = {}
    matched_types = {
        kind: 0 for kind in ("extractive", "abstractive", "boolean", "none")
    }
    successful_reference_judgments = 0
    reference_judgment_errors = 0
    non_first_winners = 0
    expected_reference_judgments = 0
    for case_index, answer in enumerate(generation_records, start=1):
        score = selected_by_case[(answer["paper_id"], answer["question_id"])]
        (
            reference_count,
            reference_successes,
            reference_errors,
            matched_type,
            non_first,
        ) = validate_v14_reference_record(
            score,
            answer,
            metrics["threshold"],
            f"DeepEval selected score {case_index}",
        )
        key = str(reference_count)
        reference_distribution[key] = reference_distribution.get(key, 0) + 1
        if answer["status"] == "success":
            expected_reference_judgments += reference_count
        successful_reference_judgments += reference_successes
        reference_judgment_errors += reference_errors
        non_first_winners += non_first
        if matched_type is not None:
            matched_types[matched_type] += 1

    reference_metrics = {
        "expected_reference_judgment_count": expected_reference_judgments,
        "successful_reference_judgment_count": successful_reference_judgments,
        "reference_judgment_error_count": reference_judgment_errors,
        "reference_count_distribution": dict(sorted(reference_distribution.items())),
        "cases_matched_by_answer_type": matched_types,
        "non_first_winning_reference_count": non_first_winners,
    }
    for field, expected in reference_metrics.items():
        if metrics.get(field) != expected:
            raise PipelineError(f"DeepEval reference aggregate {field} is contradictory.")
    recorded_reliability = metrics.get("transport_reliability")
    if not isinstance(recorded_reliability, dict):
        raise PipelineError("DeepEval transport reliability metrics are missing.")
    for field, expected in reliability.items():
        actual = recorded_reliability.get(field)
        if isinstance(expected, float):
            finite_number(actual, f"DeepEval reliability {field}", 0.0)
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6):
                raise PipelineError(
                    f"DeepEval reliability {field} is contradictory."
                )
        elif actual != expected:
            raise PipelineError(
                f"DeepEval reliability {field} is contradictory."
            )
    manifest_counts = manifest.get("counts", {})
    manifest_expected = {
        **{
            key: value
            for key, value in reliability.items()
            if key
            not in {
                "historical_judge_error_attempt_count",
                "unresolved_judge_error_count",
            }
        },
        "judge_error_attempt_count": reliability[
            "historical_judge_error_attempt_count"
        ],
        "unresolved_judge_error_count": reliability[
            "unresolved_judge_error_count"
        ],
        **reference_metrics,
    }
    for field, expected in manifest_expected.items():
        actual = manifest_counts.get(field)
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or not math.isclose(
                actual, expected, rel_tol=0.0, abs_tol=1e-6
            ):
                raise PipelineError(
                    f"DeepEval manifest reliability {field} is contradictory."
                )
        elif actual != expected:
            raise PipelineError(
                f"DeepEval manifest reliability {field} is contradictory."
            )
    return {
        "manifest": manifest,
        "metrics": metrics,
        "status": status,
        "artifacts": [
            serialize_artifact_path(manifest_path),
            serialize_artifact_path(metrics_path),
            serialize_artifact_path(scores_path),
        ],
    }


def git_provenance():
    return shared_git_provenance()


def generation_command(args):
    command = [
        sys.executable,
        "-m",
        "benchmarking.generate_answers",
        "--run-id",
        args.run_id,
    ]
    if args.max_papers is not None:
        command.extend(["--max-papers", str(args.max_papers)])
    if args.max_questions is not None:
        command.extend(["--max-questions", str(args.max_questions)])
    if args.resume_generation:
        command.append("--resume")
    if getattr(args, "select_below_threshold_from_run", None):
        command.extend(
            [
                "--select-below-threshold-from-run",
                args.select_below_threshold_from_run,
                "--selection-judge-id",
                args.selection_judge_id,
            ]
        )
    return command


def qasper_command(args):
    return [
        sys.executable,
        "-m",
        "benchmarking.evaluate_qasper_f1",
        "--run-id",
        args.run_id,
    ]


def deepeval_command(args, judge_id):
    command = [
        sys.executable,
        "-m",
        "benchmarking.evaluate_deepeval",
        "--run-id",
        args.run_id,
        "--judge-id",
        judge_id,
        "--judge-model",
        args.judge_model,
        "--threshold",
        str(args.judge_threshold),
        "--reasoning-effort",
        args.judge_reasoning_effort,
        "--max-retries",
        str(args.judge_max_retries),
    ]
    if args.judge_max_questions is not None:
        command.extend(["--max-questions", str(args.judge_max_questions)])
    if args.resume_judge:
        command.append("--resume")
    return command


def report_command(args, judge_id):
    command = [
        sys.executable,
        "-m",
        "benchmarking.generate_report",
        "--run-id",
        args.run_id,
    ]
    if args.skip_qasper:
        command.append("--skip-qasper")
    if args.skip_deepeval:
        command.append("--skip-deepeval")
    else:
        command.extend(["--judge-id", judge_id])
    return command


def validate_report_artifact(run_dir, run_id):
    report_path = run_dir / "evaluation_report.md"
    if not report_path.is_file():
        raise PipelineError("Report stage did not create evaluation_report.md.")
    content = report_path.read_text(encoding="utf-8")
    if (
        not content.startswith("# QASPER RAG Evaluation Report\n")
        or f"**Run ID:** {run_id}" not in content
    ):
        raise PipelineError("Report artifact is malformed or belongs to another run.")
    return {"path": serialize_artifact_path(report_path)}


def stage_record(
    name,
    action,
    command,
):
    return {
        "stage": name,
        "status": "pending",
        "action": action,
        "command_arguments": (
            portable_command(command)
            if command
            else None
        ),
        "started_at": None,
        "completed_at": None,
        "elapsed_seconds": None,
        "subprocess_return_code": None,
        "validated_artifact_paths": [],
        "error": None,
    }


def invocation_configuration(args, judge_id):
    return {
        "max_papers": args.max_papers,
        "max_questions": args.max_questions,
        "resume_generation": args.resume_generation,
        "skip_generation": args.skip_generation,
        "skip_qasper": args.skip_qasper,
        "skip_deepeval": args.skip_deepeval,
        "skip_report": getattr(args, "skip_report", False),
        "judge_id": judge_id,
        "judge_max_questions": args.judge_max_questions,
        "resume_judge": args.resume_judge,
        "dry_run": args.dry_run,
        "select_below_threshold_from_run": getattr(
            args, "select_below_threshold_from_run", None
        ),
        "selection_judge_id": getattr(args, "selection_judge_id", None),
    }


def build_manifest(
    args,
    judge_id,
    stages,
    started_at,
    generation_profile,
    effective_judge_profile,
    case_selection,
):
    commit_hash, dirty_state = git_provenance()
    return {
        "pipeline_version": PIPELINE_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "elapsed_seconds": None,
        "path_base": PATH_BASE,
        "generation_profile": profile_provenance(generation_profile),
        "effective_judge_profile": profile_provenance(
            effective_judge_profile
        ),
        "semantic_evaluation_contract": SEMANTIC_EVALUATION_CONTRACT,
        "case_selection": case_selection,
        "models": {"judge": effective_judge_profile.judge_model},
        "invocation_configuration": invocation_configuration(args, judge_id),
        "python_version": sys.version,
        "git_commit": commit_hash,
        "git_dirty": dirty_state,
        "stages": stages,
        "error": None,
    }


def write_pipeline_artifacts(run_dir, manifest, summary=None):
    if not run_dir.is_dir():
        return False
    try:
        run_dir.resolve().relative_to(RUNS_DIR.resolve())
    except (OSError, ValueError):
        return False
    atomic_write_json(run_dir / "pipeline_manifest.json", manifest)
    if summary is not None:
        atomic_write_json(run_dir / "pipeline_summary.json", summary)
    return True


def execute_stage(record, command):
    record["status"] = "running"
    record["started_at"] = utc_now()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            shell=False,
            check=False,
        )
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        record["error"] = "Interrupted by user."
        raise
    finally:
        record["completed_at"] = utc_now()
        record["elapsed_seconds"] = time.monotonic() - started
    record["subprocess_return_code"] = completed.returncode
    if completed.returncode != 0:
        record["status"] = "failed"
        raise PipelineError(
            f"Stage {record['stage']} exited with code {completed.returncode}."
        )


def validate_stage_output(record, validator, *arguments):
    started = time.monotonic()
    try:
        return validator(*arguments)
    finally:
        record["elapsed_seconds"] += time.monotonic() - started
        record["completed_at"] = utc_now()


def build_summary(
    manifest,
    generation=None,
    qasper=None,
    deepeval=None,
    report=None,
    failure=None,
):
    qasper_metrics = qasper["metrics"] if qasper else {}
    judge_metrics = deepeval["metrics"] if deepeval else {}
    judge_manifest = deepeval["manifest"] if deepeval else {}
    paths = {
        "pipeline_manifest": serialize_artifact_path(
            RUNS_DIR / manifest["run_id"] / "pipeline_manifest.json"
        ),
        "pipeline_summary": serialize_artifact_path(
            RUNS_DIR / manifest["run_id"] / "pipeline_summary.json"
        ),
        "generation_manifest": None,
        "answers": None,
        "qasper_metrics": None,
        "qasper_scores": None,
        "deepeval_judge_manifest": None,
        "deepeval_metrics": None,
        "deepeval_scores": None,
        "evaluation_report": report["path"] if report else None,
    }
    if generation:
        paths["generation_manifest"], paths["answers"] = generation["artifacts"]
    if qasper:
        paths["qasper_metrics"], paths["qasper_scores"] = qasper["artifacts"]
    if deepeval:
        (
            paths["deepeval_judge_manifest"],
            paths["deepeval_metrics"],
            paths["deepeval_scores"],
        ) = deepeval["artifacts"]

    qasper_skipped = next(
        stage for stage in manifest["stages"] if stage["stage"] == "qasper"
    )["action"] == "skipped"
    deepeval_skipped = next(
        stage for stage in manifest["stages"] if stage["stage"] == "deepeval"
    )["action"] == "skipped"
    return {
        "run_id": manifest["run_id"],
        "pipeline_version": PIPELINE_VERSION,
        "status": manifest["status"],
        "path_base": PATH_BASE,
        "case_selection": manifest.get("case_selection"),
        "semantic_evaluation_contract": manifest.get(
            "semantic_evaluation_contract"
        ),
        "generation_profile": manifest["generation_profile"],
        "effective_judge_profile": manifest["effective_judge_profile"],
        "models": (
            deepeval.get("metrics", {}).get("models")
            if deepeval
            else generation.get("manifest", {}).get("models")
            if generation
            else None
        ),
        "generation": {
            "status": generation["manifest"]["status"] if generation else None,
            "selected_questions": generation["selected_count"] if generation else None,
            "successful_predictions": generation["successful_count"] if generation else None,
            "generation_errors": generation["generation_error_count"] if generation else None,
            "checker_diagnostics": (
                generation["manifest"].get("checker_diagnostics")
                if generation
                else None
            ),
            "generation_case_pacing": (
                generation["manifest"].get("generation_pacing")
                if generation
                else None
            ),
        },
        "deterministic_qasper": {
            "status": "skipped" if qasper_skipped else ("complete" if qasper else None),
            "answer_f1": qasper_metrics.get("answer_f1"),
            "answer_f1_percentage": qasper_metrics.get("answer_f1_percentage"),
            "answer_f1_by_matched_answer_type": qasper_metrics.get(
                "answer_f1_by_matched_answer_type"
            ),
            "evaluated_cases": qasper_metrics.get("unique_evaluated_case_count"),
            "duplicate_attempts": qasper_metrics.get("duplicate_attempt_count"),
            "evidence_f1": qasper_metrics.get("evidence_f1"),
            "evidence_f1_note": qasper_metrics.get("evidence_f1_note"),
        },
        "deepeval": {
            "status": "skipped" if deepeval_skipped else judge_manifest.get("status"),
            "judge_id": judge_metrics.get("judge_id"),
            "model": judge_metrics.get("judge_model"),
            "evaluator_version": judge_metrics.get("evaluator_version"),
            "mean_semantic_correctness": judge_metrics.get(
                "mean_semantic_correctness"
            ),
            "full_run_semantic_correctness": judge_metrics.get(
                "full_run_semantic_correctness"
            ),
            "pass_rate": judge_metrics.get("pass_rate"),
            "threshold": judge_metrics.get("threshold"),
            "evaluation_coverage": judge_metrics.get("evaluation_coverage"),
            "successful_judgments": judge_metrics.get(
                "successful_judgment_count"
            ),
            "generation_errors": judge_metrics.get("generation_error_count"),
            "judge_errors": judge_metrics.get("judge_error_count"),
            "expected_reference_judgments": judge_metrics.get(
                "expected_reference_judgment_count"
            ),
            "successful_reference_judgments": judge_metrics.get(
                "successful_reference_judgment_count"
            ),
            "reference_judgment_errors": judge_metrics.get(
                "reference_judgment_error_count"
            ),
            "reference_count_distribution": judge_metrics.get(
                "reference_count_distribution"
            ),
            "cases_matched_by_answer_type": judge_metrics.get(
                "cases_matched_by_answer_type"
            ),
            "non_first_winning_reference_count": judge_metrics.get(
                "non_first_winning_reference_count"
            ),
            "reliability": judge_metrics.get("transport_reliability"),
            "nondeterminism_note": judge_metrics.get("nondeterminism_note"),
        },
        "report": {
            "status": next(
                stage for stage in manifest["stages"] if stage["stage"] == "report"
            )["status"],
            "path": report["path"] if report else None,
        },
        "paths": paths,
        "completion_timestamp": manifest["completed_at"],
        "error": failure,
    }


def validate_arguments(args, parser):
    if not RUN_ID_RE.fullmatch(args.run_id):
        parser.error("--run-id must be a safe run directory name")
    if args.skip_generation and args.resume_generation:
        parser.error("--skip-generation cannot be used with --resume-generation")
    if args.skip_generation and (
        args.max_papers is not None or args.max_questions is not None
    ):
        parser.error(
            "generation limits cannot be used when --skip-generation is set"
        )
    if args.skip_deepeval and args.resume_judge:
        parser.error("--skip-deepeval cannot be used with --resume-judge")
    if args.skip_deepeval and args.judge_max_questions is not None:
        parser.error(
            "--judge-max-questions has no effect with --skip-deepeval"
        )
    if args.skip_deepeval and args.judge_id is not None:
        parser.error("--judge-id has no effect with --skip-deepeval")
    if args.skip_deepeval and args.judge_reasoning_effort is not None:
        parser.error(
            "--judge-reasoning-effort has no effect with --skip-deepeval"
        )
    source_run_id = getattr(args, "select_below_threshold_from_run", None)
    source_judge_id = getattr(args, "selection_judge_id", None)
    if bool(source_run_id) != bool(source_judge_id):
        parser.error(
            "--select-below-threshold-from-run and --selection-judge-id "
            "must be supplied together"
        )
    if source_run_id and (
        args.max_papers is not None or args.max_questions is not None
    ):
        parser.error(
            "--max-papers and --max-questions cannot be combined with "
            "failure-subset selection"
        )
    if source_run_id == args.run_id:
        parser.error("the source run and new run IDs must differ")


def planned_stages(
    args,
    judge_id,
):
    return [
        stage_record(
            "generation",
            "reused" if args.skip_generation else "executed",
            (
                None
                if args.skip_generation
                else generation_command(args)
            ),
        ),
        stage_record(
            "qasper",
            "skipped" if args.skip_qasper else "executed",
            (
                None
                if args.skip_qasper
                else qasper_command(args)
            ),
        ),
        stage_record(
            "deepeval",
            "skipped" if args.skip_deepeval else "executed",
            (
                None
                if args.skip_deepeval
                else deepeval_command(args, judge_id)
            ),
        ),
        stage_record(
            "report",
            "skipped" if getattr(args, "skip_report", False) else "executed",
            None if getattr(args, "skip_report", False) else report_command(args, judge_id),
        ),
    ]


def print_dry_run(
    args,
    judge_id,
    stages,
    generation_profile,
    effective_judge_profile,
    case_selection,
):
    payload = {
        "dry_run": True,
        "run_id": args.run_id,
        "path_base": PATH_BASE,
        "generation_profile": profile_provenance(
            generation_profile,
            include_git=False,
        ),
        "effective_judge_profile": profile_provenance(
            effective_judge_profile,
            include_git=False,
        ),
        "judge_id": judge_id,
        "semantic_evaluation_contract": SEMANTIC_EVALUATION_CONTRACT,
        "case_selection": case_selection,
        "stages": [
            {
                "stage": stage["stage"],
                "action": stage["action"],
                "command_arguments": stage["command_arguments"],
            }
            for stage in stages
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def resolve_pipeline_profiles(args, run_dir):
    generation_profile = load_evaluation_config()
    requested_model = getattr(args, "judge_model", None)
    judge_model = requested_model or generation_profile.judge_model
    provisional_judge_id = args.judge_id or safe_judge_id(judge_model)
    judge_dir = run_dir / "deepeval" / provisional_judge_id
    judge_manifest_path = judge_dir / "judge_manifest.json"
    judge_snapshot = judge_dir / LEGACY_RESOLVED_CONFIG_FILENAME
    if args.resume_judge and (
        judge_manifest_path.is_file() or judge_snapshot.is_file()
    ):
        if judge_manifest_path.is_file():
            with judge_manifest_path.open("r", encoding="utf-8") as handle:
                judge_manifest = json.load(handle)
            if judge_manifest.get("evaluator_version") != DEEPEVAL_VERSION:
                raise PipelineError(
                    "Judge resume uses an incompatible evaluator/artifact "
                    "version. Start v1.6 with a new judge ID."
                )
            if judge_manifest.get("evaluation_profile") is not None:
                effective_judge_profile = config_from_provenance(
                    judge_manifest
                )
            elif judge_snapshot.is_file():
                effective_judge_profile = load_legacy_evaluation_config(
                    judge_snapshot
                )
            else:
                raise PipelineError(
                    "Judge resume lacks embedded configuration provenance."
                )
        else:
            effective_judge_profile = load_legacy_evaluation_config(
                judge_snapshot
            )
        validate_generation_profile_scope(
            effective_judge_profile,
            generation_profile,
        )
        explicit_fields = {
            "judge_model": requested_model,
            "judge_threshold": getattr(args, "judge_threshold", None),
            "judge_reasoning_effort": getattr(
                args, "judge_reasoning_effort", None
            ),
            "judge_max_retries": getattr(args, "judge_max_retries", None),
        }
        mismatches = {
            field: (getattr(effective_judge_profile, field), value)
            for field, value in explicit_fields.items()
            if value is not None
            and getattr(effective_judge_profile, field) != value
        }
        if mismatches:
            raise PipelineError(
                "Judge resume overrides do not match the judge-specific "
                f"resolved configuration: {mismatches}."
            )
    else:
        reasoning_effort = getattr(args, "judge_reasoning_effort", None)
        if (
            reasoning_effort is None
            and requested_model is not None
            and requested_model != generation_profile.judge_model
        ):
            reasoning_effort = default_reasoning_effort(
                judge_model,
                fallback=generation_profile.judge_reasoning_effort,
            )
        effective_judge_profile = generation_profile.with_judge_overrides(
            model=judge_model,
            threshold=getattr(args, "judge_threshold", None),
            reasoning_effort=reasoning_effort,
            max_retries=getattr(args, "judge_max_retries", None),
        )
    args.judge_model = effective_judge_profile.judge_model
    args.judge_threshold = effective_judge_profile.judge_threshold
    args.judge_reasoning_effort = (
        effective_judge_profile.judge_reasoning_effort
    )
    args.judge_max_retries = effective_judge_profile.judge_max_retries
    return generation_profile, effective_judge_profile


def run_pipeline(args):
    run_dir = RUNS_DIR / args.run_id
    generation_profile, effective_judge_profile = resolve_pipeline_profiles(
        args,
        run_dir,
    )
    judge_id = args.judge_id or safe_judge_id(args.judge_model)
    if not RUN_ID_RE.fullmatch(judge_id):
        raise PipelineError("Invalid judge ID.")
    case_selection = None
    if getattr(args, "select_below_threshold_from_run", None):
        try:
            selection = load_below_threshold_selection(
                args.select_below_threshold_from_run,
                args.selection_judge_id,
                generation_profile.dataset_sha256,
                generation_profile.qa_path,
            )
        except CaseSelectionError as error:
            raise PipelineError(str(error)) from error
        case_selection = selection.provenance

    if args.dry_run:
        stages = planned_stages(
            args,
            judge_id,
        )
        print_dry_run(
            args,
            judge_id,
            stages,
            generation_profile,
            effective_judge_profile,
            case_selection,
        )
        return None

    recovered = recover_stale_pipeline_manifest(
        run_dir,
        args.run_id,
        generation_profile,
        effective_judge_profile,
        case_selection,
    )
    if recovered is not None:
        print(
            f"Recovered completed pipeline {args.run_id}; summary rebuilt at "
            f"{run_dir / 'pipeline_summary.json'}"
        )
        return recovered

    return _run_pipeline(
        args,
        generation_profile,
        effective_judge_profile,
        judge_id,
        case_selection,
    )


def _run_pipeline(
    args,
    generation_profile,
    effective_judge_profile,
    judge_id,
    case_selection,
):
    stages = planned_stages(
        args,
        judge_id,
    )

    run_dir = RUNS_DIR / args.run_id
    judge_dir = run_dir / "deepeval" / judge_id
    if not args.skip_deepeval and judge_dir.exists() and not args.resume_judge:
        raise PipelineError(
            f"Judge directory already exists; use --resume-judge or a new "
            f"--judge-id: {judge_dir}"
        )

    started_at = utc_now()
    started = time.monotonic()
    manifest = build_manifest(
        args,
        judge_id,
        stages,
        started_at,
        generation_profile,
        effective_judge_profile,
        case_selection,
    )
    generation = None
    qasper = None
    deepeval = None
    report = None
    current_stage = None

    try:
        generation_stage = stages[0]
        current_stage = generation_stage
        if args.skip_generation:
            generation_stage["status"] = "validating"
            generation_stage["started_at"] = utc_now()
            stage_started = time.monotonic()
            generation = validate_generation_artifacts(
                run_dir,
                args.run_id,
                profile=generation_profile,
                require_profile=True,
                expected_selection=case_selection,
            )
            generation_stage["completed_at"] = utc_now()
            generation_stage["elapsed_seconds"] = time.monotonic() - stage_started
            generation_stage["status"] = generation["manifest"]["status"]
            generation_stage["validated_artifact_paths"] = generation["artifacts"]
        else:
            execute_stage(
                generation_stage,
                generation_command(args),
            )
            write_pipeline_artifacts(run_dir, manifest)
            generation = validate_stage_output(
                generation_stage,
                validate_generation_artifacts,
                run_dir,
                args.run_id,
                generation_profile,
                True,
                case_selection,
            )
            generation_stage["status"] = generation["manifest"]["status"]
            generation_stage["validated_artifact_paths"] = generation["artifacts"]
        manifest["models"].update(
            generation["manifest"].get("models") or {}
        )
        write_pipeline_artifacts(run_dir, manifest)

        qasper_stage = stages[1]
        current_stage = qasper_stage
        if args.skip_qasper:
            qasper_stage["status"] = "skipped"
            qasper_stage["started_at"] = utc_now()
            qasper_stage["completed_at"] = qasper_stage["started_at"]
            qasper_stage["elapsed_seconds"] = 0.0
        else:
            execute_stage(
                qasper_stage,
                qasper_command(args),
            )
            qasper = validate_stage_output(
                qasper_stage,
                validate_qasper_artifacts,
                run_dir,
                args.run_id,
                generation,
                generation_profile,
            )
            qasper_stage["status"] = "complete"
            qasper_stage["validated_artifact_paths"] = qasper["artifacts"]
        write_pipeline_artifacts(run_dir, manifest)

        deepeval_stage = stages[2]
        current_stage = deepeval_stage
        if args.skip_deepeval:
            deepeval_stage["status"] = "skipped"
            deepeval_stage["started_at"] = utc_now()
            deepeval_stage["completed_at"] = deepeval_stage["started_at"]
            deepeval_stage["elapsed_seconds"] = 0.0
        else:
            execute_stage(
                deepeval_stage,
                deepeval_command(args, judge_id),
            )
            deepeval = validate_stage_output(
                deepeval_stage,
                validate_deepeval_artifacts,
                run_dir,
                args.run_id,
                judge_id,
                args.judge_model,
                generation,
                args.judge_max_questions,
                effective_judge_profile,
            )
            deepeval_stage["status"] = deepeval["status"]
            deepeval_stage["validated_artifact_paths"] = deepeval["artifacts"]

        has_errors = generation["manifest"]["status"] == "complete_with_errors"
        if deepeval and deepeval["status"] == "complete_with_judge_errors":
            has_errors = True
        manifest["status"] = "complete_with_errors" if has_errors else "complete"

        report_stage = stages[3]
        current_stage = report_stage
        if getattr(args, "skip_report", False):
            report_stage["status"] = "skipped"
            report_stage["started_at"] = utc_now()
            report_stage["completed_at"] = report_stage["started_at"]
            report_stage["elapsed_seconds"] = 0.0
        else:
            # Freeze the completion timestamp before rendering so regenerating
            # from the final source artifacts is byte-identical.
            manifest["completed_at"] = utc_now()
            manifest["elapsed_seconds"] = time.monotonic() - started
            pre_report_summary = build_summary(
                manifest,
                generation=generation,
                qasper=qasper,
                deepeval=deepeval,
            )
            write_pipeline_artifacts(run_dir, manifest, pre_report_summary)
            execute_stage(report_stage, report_command(args, judge_id))
            report = validate_stage_output(
                report_stage,
                validate_report_artifact,
                run_dir,
                args.run_id,
            )
            report_stage["status"] = "complete"
            report_stage["validated_artifact_paths"] = [report["path"]]
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["error"] = "Interrupted by user."
        if current_stage and current_stage["status"] not in {
            "complete",
            "complete_with_errors",
            "complete_with_judge_errors",
            "skipped",
        }:
            current_stage["status"] = "interrupted"
            current_stage["error"] = "Interrupted by user."
        for stage in stages:
            if stage["status"] == "pending":
                stage["status"] = "not_run"
                stage["error"] = "Not run because the pipeline was interrupted."
        raise
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = concise_error(exc)
        if current_stage and current_stage["status"] not in {
            "complete",
            "complete_with_errors",
            "complete_with_judge_errors",
            "skipped",
        }:
            current_stage["status"] = "failed"
            current_stage["error"] = concise_error(exc)
        for stage in stages:
            if stage["status"] == "pending":
                stage["status"] = "not_run"
                stage["error"] = "Not run because an earlier stage failed."
        raise
    finally:
        if manifest["status"] in {"failed", "interrupted"}:
            manifest["completed_at"] = utc_now()
            manifest["elapsed_seconds"] = time.monotonic() - started
            summary = build_summary(
                manifest,
                generation=generation,
                qasper=qasper,
                deepeval=deepeval,
                report=report,
                failure=manifest["error"],
            )
            write_pipeline_artifacts(run_dir, manifest, summary)

    if manifest["completed_at"] is None:
        manifest["completed_at"] = utc_now()
    manifest["elapsed_seconds"] = time.monotonic() - started
    summary = build_summary(
        manifest,
        generation=generation,
        qasper=qasper,
        deepeval=deepeval,
        report=report,
    )
    write_pipeline_artifacts(run_dir, manifest, summary)
    validate_pipeline_report_agreement(run_dir, manifest, summary)
    print(
        f"Pipeline {args.run_id} finished with status {manifest['status']}. "
        f"Summary: {run_dir / 'pipeline_summary.json'}"
    )
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run generation, deterministic QASPER Answer F1, and DeepEval "
            "semantic evaluation as isolated subprocess stages, then render "
            "the deterministic Markdown report."
        )
    )
    parser.add_argument("--run-id", required=True, help="Generation run ID.")
    parser.add_argument(
        "--max-papers",
        type=positive_int,
        help="Forward a paper limit only to answer generation.",
    )
    parser.add_argument(
        "--max-questions",
        type=positive_int,
        help="Forward a total question limit only to answer generation.",
    )
    parser.add_argument(
        "--select-below-threshold-from-run",
        help=(
            "Regenerate only cases whose selected semantic score is below "
            "the recorded threshold in this source run."
        ),
    )
    parser.add_argument(
        "--selection-judge-id",
        help="Source judge ID used for failure-subset selection.",
    )
    parser.add_argument(
        "--resume-generation",
        action="store_true",
        help="Resume Stage 1 using its existing append-only behavior.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Reuse and validate an existing completed generation run.",
    )
    parser.add_argument(
        "--skip-qasper",
        action="store_true",
        help="Skip deterministic QASPER Answer F1 evaluation.",
    )
    parser.add_argument(
        "--skip-deepeval",
        action="store_true",
        help="Skip DeepEval semantic evaluation.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip deterministic Markdown report generation.",
    )
    parser.add_argument(
        "--judge-model",
        help="Override the judge model from the Python evaluation settings.",
    )
    parser.add_argument(
        "--judge-id",
        help="Judge output ID; defaults to a safe form of --judge-model.",
    )
    parser.add_argument(
        "--judge-threshold",
        type=threshold_value,
        help="Override the judge threshold from the Python evaluation settings.",
    )
    parser.add_argument(
        "--judge-reasoning-effort",
        choices=("none", "default", "low", "medium", "high"),
        help="Optional provider reasoning effort forwarded to Stage 3.",
    )
    parser.add_argument(
        "--judge-max-retries",
        type=nonnegative_int,
        help="Override retries from the Python evaluation settings.",
    )
    parser.add_argument(
        "--judge-max-questions",
        type=positive_int,
        help="Judge only the first N generated cases; independent of generation.",
    )
    parser.add_argument(
        "--resume-judge",
        action="store_true",
        help="Resume the matching append-only Stage 3 judge run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sanitized planned commands without executing or writing.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(args, parser)
    try:
        run_pipeline(args)
    except PipelineError as exc:
        parser.exit(1, f"Pipeline failed: {concise_error(exc)}\n")


if __name__ == "__main__":
    main()
