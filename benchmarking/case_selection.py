"""Read-only, provenance-validated selection of prior semantic failures."""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from benchmarking.evaluation_config import sha256_json
from benchmarking.evaluate_qasper_f1 import read_answer_attempts


EVAL_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVAL_DIR / "runs"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMPLETED_GENERATION_STATUSES = {"complete", "complete_with_errors"}
COMPLETED_JUDGE_STATUSES = {"complete", "complete_with_judge_errors"}
SCORE_STATUSES = {"success", "generation_error", "judge_error"}


class CaseSelectionError(ValueError):
    """A source run cannot be used reproducibly for case selection."""


@dataclass(frozen=True)
class CaseSelection:
    case_keys: tuple[tuple[str, str], ...]
    provenance: dict


def _load_json(path, label):
    if not path.is_file():
        raise CaseSelectionError(f"Missing {label}.")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise CaseSelectionError(f"Malformed {label}.") from error
    if not isinstance(value, dict):
        raise CaseSelectionError(f"Malformed {label}: expected an object.")
    return value


def _load_jsonl(path, label):
    if not path.is_file():
        raise CaseSelectionError(f"Missing {label}.")
    records = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record is not an object")
                records.append((line_number, value))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise CaseSelectionError(f"Malformed {label}.") from error
    if not records:
        raise CaseSelectionError(f"Empty {label}.")
    return records


def _dataset_digest(artifact, label):
    provenance = artifact.get("evaluation_profile")
    digest = provenance.get("dataset_sha256") if isinstance(provenance, dict) else None
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise CaseSelectionError(f"{label} has invalid dataset provenance.")
    return digest


def _dataset_order(qa_path):
    try:
        with Path(qa_path).open("r", encoding="utf-8") as handle:
            papers = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise CaseSelectionError("Current QASPER dataset is unreadable.") from error
    if not isinstance(papers, list) or not papers:
        raise CaseSelectionError("Current QASPER dataset is malformed.")
    ordered = []
    seen = set()
    for paper in papers:
        if not isinstance(paper, dict) or not isinstance(paper.get("id"), str):
            raise CaseSelectionError("Current QASPER dataset has invalid papers.")
        questions = (paper.get("qas") or {}).get("question_id")
        if not isinstance(questions, list):
            raise CaseSelectionError("Current QASPER dataset has invalid questions.")
        for question_id in questions:
            key = (paper["id"], question_id)
            if (
                not isinstance(question_id, str)
                or not question_id
                or key in seen
            ):
                raise CaseSelectionError("Current QASPER dataset has duplicate cases.")
            seen.add(key)
            ordered.append(key)
    return ordered


def _select_attempt(records):
    successes = [record for record in records if record["status"] == "success"]
    generation_errors = [
        record for record in records if record["status"] == "generation_error"
    ]
    if successes:
        return successes[-1]
    if generation_errors:
        return generation_errors[-1]
    return records[-1]


def load_below_threshold_selection(
    source_run_id,
    source_judge_id,
    current_dataset_sha256,
    qa_path,
):
    """Select finite below-threshold latest results in saved dataset order."""
    if not SAFE_ID_RE.fullmatch(source_run_id or ""):
        raise CaseSelectionError("Invalid source run ID.")
    if not SAFE_ID_RE.fullmatch(source_judge_id or ""):
        raise CaseSelectionError("Invalid source judge ID.")
    if not isinstance(current_dataset_sha256, str) or not SHA256_RE.fullmatch(
        current_dataset_sha256
    ):
        raise CaseSelectionError("Invalid current dataset digest.")

    source_dir = RUNS_DIR / source_run_id
    judge_dir = source_dir / "deepeval" / source_judge_id
    generation_manifest = _load_json(
        source_dir / "generation_manifest.json", "source generation manifest"
    )
    judge_manifest = _load_json(
        judge_dir / "judge_manifest.json", "source judge manifest"
    )
    metrics = _load_json(judge_dir / "metrics.json", "source judge metrics")
    score_lines = _load_jsonl(judge_dir / "scores.jsonl", "source judge scores")

    if generation_manifest.get("run_id") != source_run_id:
        raise CaseSelectionError("Source generation manifest run ID is inconsistent.")
    if generation_manifest.get("status") not in COMPLETED_GENERATION_STATUSES:
        raise CaseSelectionError("Source generation run is not complete.")
    source_dataset_sha256 = _dataset_digest(
        generation_manifest, "Source generation manifest"
    )
    if source_dataset_sha256 != current_dataset_sha256:
        raise CaseSelectionError(
            "Source and current QASPER dataset SHA-256 digests do not match."
        )

    expected_identity = {
        "generation_run_id": source_run_id,
        "judge_id": source_judge_id,
    }
    for field, expected in expected_identity.items():
        if judge_manifest.get(field) != expected or metrics.get(field) != expected:
            raise CaseSelectionError(f"Source judge {field} is inconsistent.")
    if judge_manifest.get("status") not in COMPLETED_JUDGE_STATUSES:
        raise CaseSelectionError("Source judge run is not complete.")
    evaluator_version = judge_manifest.get("evaluator_version")
    judge_model = judge_manifest.get("judge_model")
    if (
        not isinstance(evaluator_version, str)
        or not evaluator_version
        or metrics.get("evaluator_version") != evaluator_version
        or not isinstance(judge_model, str)
        or not judge_model
        or metrics.get("judge_model") != judge_model
    ):
        raise CaseSelectionError("Source judge model or evaluator version is inconsistent.")
    threshold = metrics.get("threshold")
    manifest_threshold = (judge_manifest.get("judge_request") or {}).get("threshold")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
        or manifest_threshold != threshold
    ):
        raise CaseSelectionError("Source judge threshold is invalid or inconsistent.")
    for artifact, label in (
        (judge_manifest, "Source judge manifest"),
        (metrics, "Source judge metrics"),
    ):
        if _dataset_digest(artifact, label) != source_dataset_sha256:
            raise CaseSelectionError(f"{label} dataset provenance is inconsistent.")

    answers_path = source_dir / "answers.jsonl"
    try:
        answers, _ = read_answer_attempts(answers_path)
    except (OSError, ValueError, TypeError) as error:
        raise CaseSelectionError("Source generation answers are malformed.") from error
    answer_by_key = {
        (record["paper_id"], record["question_id"]): record for record in answers
    }
    if len(answer_by_key) != len(answers):
        raise CaseSelectionError("Source generation answers contain duplicate cases.")
    selected_questions = (generation_manifest.get("counts") or {}).get(
        "selected_questions"
    )
    if selected_questions != len(answers):
        raise CaseSelectionError("Source generation case count is inconsistent.")

    grouped = {}
    attempt_numbers = {}
    resume_key_cases = {}
    for line_number, score in score_lines:
        required = {
            "paper_id",
            "question_id",
            "judge_model",
            "judge_id",
            "evaluator_version",
            "threshold",
            "status",
            "semantic_score",
            "attempt_number",
            "resume_key",
        }
        if required - score.keys():
            raise CaseSelectionError(
                f"Source judge score line {line_number} is incomplete."
            )
        key = (score["paper_id"], score["question_id"])
        if key not in answer_by_key:
            raise CaseSelectionError("Source judge score has an unknown case identity.")
        if (
            score["judge_model"] != judge_model
            or score["judge_id"] != source_judge_id
            or score["evaluator_version"] != evaluator_version
            or score["threshold"] != threshold
            or score["status"] not in SCORE_STATUSES
        ):
            raise CaseSelectionError("Source judge score identity is inconsistent.")
        expected_resume_key = json.dumps(
            [key[0], key[1], judge_model, evaluator_version],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if score["resume_key"] != expected_resume_key:
            raise CaseSelectionError("Source judge score has an invalid resume key.")
        if expected_resume_key in resume_key_cases and resume_key_cases[
            expected_resume_key
        ] != key:
            raise CaseSelectionError("Source judge resume key is ambiguous.")
        resume_key_cases[expected_resume_key] = key
        attempt = score["attempt_number"]
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise CaseSelectionError("Source judge score has an invalid attempt number.")
        seen_attempts = attempt_numbers.setdefault(key, set())
        if attempt in seen_attempts:
            raise CaseSelectionError("Source judge score duplicates an attempt number.")
        seen_attempts.add(attempt)
        grouped.setdefault(key, []).append(score)

    selected_case_count = metrics.get("selected_case_count")
    if (
        not isinstance(selected_case_count, int)
        or isinstance(selected_case_count, bool)
        or selected_case_count < 1
        or selected_case_count != len(grouped)
        or (judge_manifest.get("counts") or {}).get("selected_cases")
        != selected_case_count
    ):
        raise CaseSelectionError("Source judge selected-case count is inconsistent.")

    selected_scores = {key: _select_attempt(records) for key, records in grouped.items()}
    successful_count = sum(
        score["status"] == "success" for score in selected_scores.values()
    )
    generation_error_count = sum(
        score["status"] == "generation_error"
        for score in selected_scores.values()
    )
    judge_error_count = sum(
        score["status"] == "judge_error" for score in selected_scores.values()
    )
    non_null_scores = [
        score["semantic_score"]
        for score in selected_scores.values()
        if score.get("semantic_score") is not None
    ]
    expected_metric_counts = {
        "successful_judgment_count": successful_count,
        "generation_error_count": generation_error_count,
        "judge_error_count": judge_error_count,
        "non_null_evaluated_case_count": len(non_null_scores),
    }
    for field, expected in expected_metric_counts.items():
        if metrics.get(field) != expected:
            raise CaseSelectionError(f"Source judge metric {field} is inconsistent.")
    expected_coverage = len(non_null_scores) / selected_case_count
    coverage = metrics.get("evaluation_coverage")
    if (
        not isinstance(coverage, (int, float))
        or isinstance(coverage, bool)
        or not math.isfinite(coverage)
        or not math.isclose(coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise CaseSelectionError("Source judge evaluation coverage is inconsistent.")
    expected_mean = (
        sum(non_null_scores) / len(non_null_scores) if non_null_scores else None
    )
    recorded_mean = metrics.get("mean_semantic_correctness")
    if expected_mean is None:
        if recorded_mean is not None:
            raise CaseSelectionError("Source judge semantic mean is inconsistent.")
    elif (
        not isinstance(recorded_mean, (int, float))
        or isinstance(recorded_mean, bool)
        or not math.isclose(
            recorded_mean, expected_mean, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise CaseSelectionError("Source judge semantic mean is inconsistent.")
    manifest_counts = judge_manifest.get("counts") or {}
    if (
        manifest_counts.get("successful_judgments") != successful_count
        or manifest_counts.get("unresolved_judge_error_count")
        != judge_error_count
    ):
        raise CaseSelectionError("Source judge manifest counts are inconsistent.")

    eligible = set()
    for key, score in selected_scores.items():
        value = score.get("semantic_score")
        if value is None:
            continue
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise CaseSelectionError("Source judge score has an invalid semantic score.")
        if value < threshold:
            eligible.add(key)

    dataset_order = _dataset_order(qa_path)
    dataset_keys = set(dataset_order)
    if not set(answer_by_key).issubset(dataset_keys) or not set(grouped).issubset(
        dataset_keys
    ):
        raise CaseSelectionError("Source cases do not match the current dataset.")
    ordered_case_keys = tuple(key for key in dataset_order if key in eligible)
    if len(ordered_case_keys) != len(eligible):
        raise CaseSelectionError("Selected source cases are ambiguous.")
    serialized_keys = [f"{paper_id}/{question_id}" for paper_id, question_id in ordered_case_keys]
    provenance = {
        "selection_mode": "prior_semantic_below_threshold",
        "source_run_id": source_run_id,
        "source_judge_id": source_judge_id,
        "source_evaluator_version": evaluator_version,
        "source_threshold": threshold,
        "selected_case_count": len(ordered_case_keys),
        "ordered_case_keys": serialized_keys,
        "ordered_case_keys_sha256": sha256_json(serialized_keys),
        "source_dataset_sha256": source_dataset_sha256,
    }
    return CaseSelection(case_keys=ordered_case_keys, provenance=provenance)
