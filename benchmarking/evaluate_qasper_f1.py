"""Score generated answers with deterministic QASPER Answer F1 metrics.

Portions are adapted from AllenAI's official QASPER evaluator:
https://github.com/allenai/qasper-led-baseline/blob/e996b6c7b1b5f95d9308a74e3586416c6e780df1/scripts/evaluator.py
The upstream material is licensed under Apache License 2.0; this file contains
Measured RAG modifications licensed under AGPL-3.0-only. See LICENSE,
THIRD_PARTY_NOTICES.md, and third_party/licenses/Apache-2.0.txt.
"""

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from benchmarking.artifact_io import atomic_write_json, atomic_write_text
from benchmarking.evaluation_config import (
    PATH_BASE,
    load_evaluation_config,
    serialize_artifact_path,
    validate_generation_profile_provenance,
)

EVAL_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVAL_DIR / "runs"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
STANDARD_ABSTENTION = "The document does not contain this information."
ACCEPTED_GENERATION_STATUSES = {"complete", "complete_with_errors"}
OFFICIAL_ANSWER_TYPES = ("extractive", "abstractive", "boolean", "none")
OFFICIAL_EVALUATOR_URL = (
    "https://github.com/allenai/qasper-led-baseline/blob/e996b6c7b1b5f95d9308a74e3586416c6e780df1/"
    "scripts/evaluator.py"
)
REQUIRED_RECORD_FIELDS = {
    "paper_id",
    "question_id",
    "question",
    "gold_answers",
    "conversation_id",
    "raw_answer",
    "cleaned_answer",
    "status",
    "error",
    "elapsed_seconds",
}
REQUIRED_ANNOTATION_FIELDS = {
    "unanswerable",
    "extractive_spans",
    "yes_no",
    "free_form_answer",
    "evidence",
    "highlighted_evidence",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize_answer(text):
    if not isinstance(text, str):
        raise TypeError("Answer normalization requires a string.")

    lowered = text.lower()
    without_punctuation = "".join(
        character
        for character in lowered
        if character not in string.punctuation
    )
    without_articles = ARTICLES_RE.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def token_f1(prediction, reference):
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    matching_tokens = sum(common.values())
    if matching_tokens == 0:
        return 0.0

    precision = matching_tokens / len(prediction_tokens)
    recall = matching_tokens / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def canonicalize_prediction(prediction):
    if not isinstance(prediction, str):
        return ""
    if STANDARD_ABSTENTION.lower() in prediction.lower():
        return "Unanswerable"
    return prediction


def convert_annotation(annotation):
    if not isinstance(annotation, dict):
        raise ValueError("Gold annotation must be an object.")

    missing = REQUIRED_ANNOTATION_FIELDS - annotation.keys()
    if missing:
        raise ValueError(
            "Gold annotation is missing fields: "
            + ", ".join(sorted(missing))
        )

    unanswerable = annotation["unanswerable"]
    extractive_spans = annotation["extractive_spans"]
    yes_no = annotation["yes_no"]
    free_form_answer = annotation["free_form_answer"]

    if not isinstance(unanswerable, bool):
        raise ValueError("Gold annotation unanswerable must be boolean.")
    if (
        not isinstance(extractive_spans, list)
        or any(not isinstance(span, str) for span in extractive_spans)
    ):
        raise ValueError("Gold annotation extractive_spans must be strings.")
    if not isinstance(free_form_answer, str):
        raise ValueError("Gold annotation free_form_answer must be a string.")
    if yes_no is not None and not isinstance(yes_no, bool):
        raise ValueError("Gold annotation yes_no must be boolean or null.")
    for evidence_field in ("evidence", "highlighted_evidence"):
        evidence = annotation[evidence_field]
        if (
            not isinstance(evidence, list)
            or any(not isinstance(item, str) for item in evidence)
        ):
            raise ValueError(
                f"Gold annotation {evidence_field} must be a list of strings."
            )

    if unanswerable:
        return "Unanswerable", "none"
    if extractive_spans:
        return ", ".join(extractive_spans), "extractive"
    if free_form_answer:
        return free_form_answer, "abstractive"
    if yes_no is True:
        return "Yes", "boolean"
    if yes_no is False:
        return "No", "boolean"

    raise ValueError("Gold annotation does not contain a valid answer.")


def validate_gold_bundle(bundle, case_label):
    if not isinstance(bundle, dict):
        raise ValueError(f"{case_label}: gold_answers must be an object.")

    required = ("answer", "annotation_id", "worker_id")
    missing = [field for field in required if field not in bundle]
    if missing:
        raise ValueError(
            f"{case_label}: gold_answers is missing fields: "
            + ", ".join(missing)
        )

    annotations = bundle["answer"]
    annotation_ids = bundle["annotation_id"]
    worker_ids = bundle["worker_id"]
    if (
        not isinstance(annotations, list)
        or not annotations
        or not isinstance(annotation_ids, list)
        or not isinstance(worker_ids, list)
        or len(annotations) != len(annotation_ids)
        or len(annotations) != len(worker_ids)
    ):
        raise ValueError(f"{case_label}: gold annotation arrays are misaligned.")

    converted = []
    for index, annotation in enumerate(annotations):
        answer, answer_type = convert_annotation(annotation)
        annotation_id = annotation_ids[index]
        worker_id = worker_ids[index]
        if not isinstance(annotation_id, str) or not annotation_id:
            raise ValueError(f"{case_label}: invalid annotation_id at index {index}.")
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError(f"{case_label}: invalid worker_id at index {index}.")
        converted.append(
            {
                "annotation_index": index,
                "annotation_id": annotation_id,
                "worker_id": worker_id,
                "answer": answer,
                "answer_type": answer_type,
            }
        )

    return converted


def validate_record(record, line_number):
    if not isinstance(record, dict):
        raise ValueError(f"answers.jsonl line {line_number} must be an object.")

    missing = REQUIRED_RECORD_FIELDS - record.keys()
    if missing:
        raise ValueError(
            f"answers.jsonl line {line_number} is missing fields: "
            + ", ".join(sorted(missing))
        )

    for field in ("paper_id", "question_id", "question", "conversation_id"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(
                f"answers.jsonl line {line_number} has invalid {field}."
            )

    status = record["status"]
    if status not in {"success", "error"}:
        raise ValueError(
            f"answers.jsonl line {line_number} has invalid status: {status!r}."
        )
    if status == "success" and (
        not isinstance(record["cleaned_answer"], str)
        or not record["cleaned_answer"].strip()
    ):
        raise ValueError(
            f"answers.jsonl line {line_number} is successful but has no "
            "non-empty cleaned_answer."
        )
    if record["cleaned_answer"] is not None and not isinstance(
        record["cleaned_answer"], str
    ):
        raise ValueError(
            f"answers.jsonl line {line_number} has invalid cleaned_answer."
        )
    if record["error"] is not None and not isinstance(record["error"], str):
        raise ValueError(f"answers.jsonl line {line_number} has invalid error.")
    if record["raw_answer"] is not None and not isinstance(
        record["raw_answer"], str
    ):
        raise ValueError(
            f"answers.jsonl line {line_number} has invalid raw_answer."
        )
    if (
        not isinstance(record["elapsed_seconds"], (int, float))
        or isinstance(record["elapsed_seconds"], bool)
        or record["elapsed_seconds"] < 0
    ):
        raise ValueError(
            f"answers.jsonl line {line_number} has invalid elapsed_seconds."
        )

    case_label = f"{record['paper_id']}/{record['question_id']}"
    validate_gold_bundle(record["gold_answers"], case_label)


def read_answer_attempts(path):
    if not path.is_file():
        raise FileNotFoundError(f"Answers file not found: {path}")

    attempts_by_case = {}
    case_order = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {path} at line {line_number}."
                ) from exc

            validate_record(record, line_number)
            key = (record["paper_id"], record["question_id"])
            if key not in attempts_by_case:
                attempts_by_case[key] = []
                case_order.append(key)
            else:
                first = attempts_by_case[key][0]
                if (
                    record["question"] != first["question"]
                    or record["gold_answers"] != first["gold_answers"]
                ):
                    raise ValueError(
                        f"Duplicate case {key[0]}/{key[1]} has inconsistent "
                        "question or gold annotations."
                    )
            attempts_by_case[key].append(record)

    if not attempts_by_case:
        raise ValueError(f"Answers file contains no records: {path}")

    selected_records = []
    for key in case_order:
        attempts = attempts_by_case[key]
        successful = [
            record for record in attempts if record["status"] == "success"
        ]
        selected_records.append(successful[-1] if successful else attempts[-1])

    duplicate_attempt_count = sum(
        len(attempts) - 1 for attempts in attempts_by_case.values()
    )
    return selected_records, duplicate_attempt_count


def score_record(record):
    case_label = f"{record['paper_id']}/{record['question_id']}"
    converted_references = validate_gold_bundle(
        record["gold_answers"],
        case_label,
    )

    original_prediction = record["cleaned_answer"]
    prediction_available = (
        record["status"] == "success"
        and isinstance(original_prediction, str)
        and bool(original_prediction.strip())
    )
    canonical_prediction = (
        canonicalize_prediction(original_prediction)
        if prediction_available
        else ""
    )

    best_reference = None
    best_f1 = -1.0
    scored_references = []
    for reference in converted_references:
        reference_f1 = (
            token_f1(canonical_prediction, reference["answer"])
            if prediction_available
            else 0.0
        )
        scored_reference = dict(reference)
        scored_reference["f1"] = reference_f1
        scored_references.append(scored_reference)
        if reference_f1 > best_f1:
            best_f1 = reference_f1
            best_reference = scored_reference

    return {
        "paper_id": record["paper_id"],
        "question_id": record["question_id"],
        "question": record["question"],
        "original_cleaned_prediction": original_prediction,
        "canonical_prediction": canonical_prediction,
        "status": record["status"],
        "references": scored_references,
        "best_reference": best_reference["answer"],
        "matched_answer_type": best_reference["answer_type"],
        "answer_f1": best_f1,
        "generation_error": record["error"],
    }


def load_manifest(path, run_id):
    if not path.is_file():
        raise FileNotFoundError(f"Generation manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Generation manifest must be an object.")
    if manifest.get("run_id") != run_id:
        raise ValueError(
            "Generation manifest run ID does not match the requested run ID."
        )
    status = manifest.get("status")
    if status not in ACCEPTED_GENERATION_STATUSES:
        raise ValueError(
            f"Generation run status {status!r} is not evaluable; expected "
            "'complete' or 'complete_with_errors'."
        )
    selected_count = manifest.get("counts", {}).get("selected_questions")
    if (
        not isinstance(selected_count, int)
        or isinstance(selected_count, bool)
        or selected_count < 1
    ):
        raise ValueError(
            "Generation manifest has no valid selected-question count."
        )
    return manifest, selected_count


def write_scores(path, scores):
    atomic_write_text(
        path,
        "".join(
            json.dumps(score, ensure_ascii=False) + "\n" for score in scores
        ),
    )


def write_metrics(path, metrics):
    atomic_write_json(path, metrics)


def evaluate_run(run_id):
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "Run ID must start with an alphanumeric character and contain "
            "only letters, numbers, dots, underscores, or hyphens."
        )

    run_dir = RUNS_DIR / run_id
    answers_path = run_dir / "answers.jsonl"
    manifest_path = run_dir / "generation_manifest.json"
    metrics_path = run_dir / "qasper_metrics.json"
    scores_path = run_dir / "qasper_scores.jsonl"
    profile = load_evaluation_config()

    generation_manifest, selected_question_count = load_manifest(
        manifest_path, run_id
    )
    validate_generation_profile_provenance(
        generation_manifest,
        profile,
        records_path=answers_path,
        allow_legacy_read=False,
    )
    selected_records, duplicate_attempt_count = read_answer_attempts(
        answers_path
    )
    unique_case_count = len(selected_records)
    if unique_case_count != selected_question_count:
        raise ValueError(
            "Manifest selected-question count does not match unique answer "
            f"cases: {selected_question_count} != {unique_case_count}."
        )

    scores = [score_record(record) for record in selected_records]
    answer_f1 = sum(score["answer_f1"] for score in scores) / len(scores)

    scores_by_type = defaultdict(list)
    for score in scores:
        scores_by_type[score["matched_answer_type"]].append(
            score["answer_f1"]
        )
    answer_f1_by_type = {
        answer_type: (
            sum(scores_by_type[answer_type])
            / len(scores_by_type[answer_type])
            if scores_by_type[answer_type]
            else 0.0
        )
        for answer_type in OFFICIAL_ANSWER_TYPES
    }

    successful_prediction_count = sum(
        score["status"] == "success" for score in scores
    )
    generation_error_count = sum(
        score["status"] == "error" for score in scores
    )
    error_missing_count = generation_error_count
    metrics = {
        "run_id": run_id,
        "path_base": PATH_BASE,
        "evaluation_profile": generation_manifest["evaluation_profile"],
        "models": generation_manifest.get("models"),
        "metric_name": "QASPER Answer F1",
        "metric_provenance": {
            "official_evaluator_url": OFFICIAL_EVALUATOR_URL,
            "implementation": (
                "Adapted from the official QASPER evaluator for this "
                "project's append-only JSONL generation runs; this is not "
                "the unmodified official script."
            ),
        },
        "normalization": (
            "Lowercase, remove ASCII punctuation, remove articles "
            "a/an/the, and normalize whitespace; token-overlap F1 uses "
            "the maximum score across human references."
        ),
        "answer_f1": answer_f1,
        "answer_f1_percentage": answer_f1 * 100,
        "answer_f1_by_matched_answer_type": answer_f1_by_type,
        "selected_question_count": selected_question_count,
        "unique_evaluated_case_count": unique_case_count,
        "successful_prediction_count": successful_prediction_count,
        "generation_error_count": generation_error_count,
        "error_missing_count": error_missing_count,
        "failure_handling_note": (
            "Recorded generation errors score zero. An actually absent "
            "selected answer record is an integrity error and stops "
            "evaluation."
        ),
        "duplicate_attempt_count": duplicate_attempt_count,
        "source_answers_path": serialize_artifact_path(answers_path),
        "evaluation_timestamp": utc_now(),
        "evidence_f1": None,
        "evidence_f1_note": (
            "Evidence F1 is not computed because retrieved chunks are not "
            "mapped to QASPER evidence format."
        ),
    }

    write_scores(scores_path, scores)
    write_metrics(metrics_path, metrics)
    print(
        f"QASPER Answer F1: {answer_f1:.10f} "
        f"({answer_f1 * 100:.8f}%) across {unique_case_count} cases."
    )
    return metrics, scores


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic QASPER Answer F1."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Generation run under benchmarking/runs/<run-id>.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    evaluate_run(args.run_id)


if __name__ == "__main__":
    main()
