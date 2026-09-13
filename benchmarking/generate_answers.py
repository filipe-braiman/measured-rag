"""Generate QASPER answers through the production RAG pipeline."""

import argparse
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from benchmarking.evaluation_config import (
    PATH_BASE,
    REPO_ROOT,
    load_evaluation_config,
    profile_provenance,
    sanitize_error as sanitize_error_text,
    serialize_artifact_path,
    validate_generation_profile_provenance,
)
from benchmarking.artifact_io import atomic_write_json, atomic_write_text
from benchmarking.checker_diagnostics import (
    aggregate_checker_diagnostics,
    empty_checker_diagnostics,
    sanitize_application_log,
)
from benchmarking.case_selection import load_below_threshold_selection

EVAL_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVAL_DIR / "runs"

DIAGNOSTIC_SUFFIX_RE = re.compile(
    # Match the complete terminal UI structure, never isolated diagnostic words.
    r"\r?\n\r?\n(?:"
    r"Retrieval relevance: (?:(?:🟢 )?High|(?:🟡 )?Medium|(?:🔴 )?Low|(?:⚪ )?Unknown)[ \t]*"
    r"\r?\nGroundedness: (?:(?:🟢 )?Grounded|(?:🟡 )?Partially grounded|(?:🔴 )?Not grounded|(?:⚪ )?Unknown)[ \t]*"
    r"(?:\r?\n(?:📚 )?Scope: (?:all indexed documents \([0-9]+\)|selected documents \([^\r\n]+\)))?"
    r"|"
    r"---[ \t]*\r?\n\r?\n\*\*Answer diagnostics\*\*[ \t]*\r?\n\r?\n"
    r"- \*\*Retrieval relevance:\*\* (?:High|Medium|Low|Unknown)[ \t]*"
    r"\r?\n- \*\*Groundedness:\*\* (?:Grounded|Partially grounded|Not grounded|Unknown)[ \t]*"
    r"(?:\r?\n- \*\*Scope:\*\* (?:All indexed documents \([0-9]+\)|Selected documents \([^\r\n]+\)))?"
    r")[ \t\r\n]*\Z"
)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
EMPTY_CLEANED_ANSWER_ERROR = (
    "GenerationIntegrityError: The provider returned no substantive answer "
    "after diagnostic stripping."
)
REQUIRED_ANSWER_RECORD_FIELDS = {
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
REQUIRED_QAS_FIELDS = (
    "question",
    "question_id",
    "nlp_background",
    "topic_background",
    "paper_read",
    "search_query",
    "question_writer",
    "answers",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def clean_answer(answer):
    if not isinstance(answer, str):
        raise TypeError("Answer must be a string.")
    return DIAGNOSTIC_SUFFIX_RE.sub("", answer)


def classify_returned_answer(raw_answer):
    cleaned_answer = clean_answer(raw_answer)
    if not cleaned_answer.strip():
        return cleaned_answer, "error", EMPTY_CLEANED_ANSWER_ERROR
    return cleaned_answer, "success", None


def case_key(paper_id, question_id):
    return str(paper_id), str(question_id)


def sanitize_error(error):
    return f"{type(error).__name__}: {sanitize_error_text(error)}"[:1000]


def load_qasper_records(path=None):
    path = load_evaluation_config().qa_path if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"QASPER QA data not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list) or not records:
        raise ValueError("QASPER QA data must be a non-empty list of papers.")

    validated = []
    seen_paper_ids = set()
    seen_cases = set()

    for paper_index, paper in enumerate(records):
        if not isinstance(paper, dict):
            raise ValueError(f"Paper {paper_index} must be an object.")

        paper_id = paper.get("id")
        qas = paper.get("qas")
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise ValueError(f"Paper {paper_index} has an invalid id.")
        if paper_id in seen_paper_ids:
            raise ValueError(f"Duplicate paper id: {paper_id}")
        if not isinstance(qas, dict):
            raise ValueError(f"Paper {paper_id} has invalid qas data.")

        missing_fields = [
            field for field in REQUIRED_QAS_FIELDS
            if field not in qas
        ]
        if missing_fields:
            raise ValueError(
                f"Paper {paper_id} is missing qas fields: "
                f"{', '.join(missing_fields)}"
            )

        questions = qas["question"]
        if not isinstance(questions, list):
            raise ValueError(f"Paper {paper_id} qas.question must be a list.")

        question_count = len(questions)
        for field in REQUIRED_QAS_FIELDS:
            values = qas[field]
            if not isinstance(values, list) or len(values) != question_count:
                raise ValueError(
                    f"Paper {paper_id} qas.{field} must contain "
                    f"{question_count} entries."
                )

        for question_index in range(question_count):
            question = qas["question"][question_index]
            question_id = qas["question_id"][question_index]
            answer_bundle = qas["answers"][question_index]

            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Paper {paper_id} question {question_index} is invalid."
                )
            if not isinstance(question_id, str) or not question_id.strip():
                raise ValueError(
                    f"Paper {paper_id} question_id {question_index} is invalid."
                )

            key = case_key(paper_id, question_id)
            if key in seen_cases:
                raise ValueError(
                    f"Duplicate QASPER case: {paper_id}/{question_id}"
                )
            seen_cases.add(key)

            _validate_answer_bundle(
                answer_bundle,
                paper_id,
                question_id,
            )

        seen_paper_ids.add(paper_id)
        validated.append(paper)

    return validated


def _validate_answer_bundle(bundle, paper_id, question_id):
    if not isinstance(bundle, dict):
        raise ValueError(
            f"Answers for {paper_id}/{question_id} must be an object."
        )

    required = ("answer", "annotation_id", "worker_id")
    if any(field not in bundle for field in required):
        raise ValueError(
            f"Answers for {paper_id}/{question_id} are incomplete."
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
        raise ValueError(
            f"Answer arrays for {paper_id}/{question_id} are misaligned."
        )

    answer_fields = {
        "unanswerable",
        "extractive_spans",
        "yes_no",
        "free_form_answer",
        "evidence",
        "highlighted_evidence",
    }
    for annotation_index, annotation in enumerate(annotations):
        if (
            not isinstance(annotation, dict)
            or not answer_fields.issubset(annotation)
        ):
            raise ValueError(
                f"Annotation {annotation_index} for "
                f"{paper_id}/{question_id} is invalid."
            )


def select_cases(records, max_papers=None, max_questions=None, case_keys=None):
    if case_keys is not None:
        requested = set(case_keys)
        grouped_cases = []
        found = set()
        for paper in records:
            indices = [
                index
                for index, question_id in enumerate(paper["qas"]["question_id"])
                if case_key(paper["id"], question_id) in requested
            ]
            if indices:
                grouped_cases.append((paper, indices))
                found.update(
                    case_key(paper["id"], paper["qas"]["question_id"][index])
                    for index in indices
                )
        if found != requested:
            raise ValueError("One or more selected case keys are absent from QASPER data.")
        return grouped_cases

    selected = records[:max_papers] if max_papers is not None else records
    grouped_cases = []
    remaining = max_questions

    for paper in selected:
        qas = paper["qas"]
        count = len(qas["question"])
        if remaining is not None:
            count = min(count, remaining)
        if count:
            grouped_cases.append((paper, list(range(count))))
        if remaining is not None:
            remaining -= count
            if remaining == 0:
                break

    return grouped_cases


def load_existing_run_records(answers_path):
    completed = set()
    conversation_ids = set()
    record_count = 0
    successful_count = 0
    failed_count = 0

    if not answers_path.exists():
        return (
            completed,
            conversation_ids,
            record_count,
            successful_count,
            failed_count,
        )

    with answers_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {answers_path} at line {line_number}."
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid answer record in {answers_path} at line "
                    f"{line_number}: expected an object."
                )
            missing = REQUIRED_ANSWER_RECORD_FIELDS - record.keys()
            if missing:
                raise ValueError(
                    f"Invalid answer record in {answers_path} at line "
                    f"{line_number}: missing {', '.join(sorted(missing))}."
                )
            for field in (
                "paper_id",
                "question_id",
                "question",
                "conversation_id",
            ):
                if not isinstance(record[field], str) or not record[field].strip():
                    raise ValueError(
                        f"Invalid answer record in {answers_path} at line "
                        f"{line_number}: {field} must be a non-empty string."
                    )
            status = record["status"]
            if status not in {"success", "error"}:
                raise ValueError(
                    f"Invalid answer record in {answers_path} at line "
                    f"{line_number}: unsupported status {status!r}."
                )
            cleaned_answer = record["cleaned_answer"]
            if cleaned_answer is not None and not isinstance(cleaned_answer, str):
                raise ValueError(
                    f"Invalid answer record in {answers_path} at line "
                    f"{line_number}: cleaned_answer must be a string or null."
                )
            if status == "success" and (
                not isinstance(cleaned_answer, str)
                or not cleaned_answer.strip()
            ):
                raise ValueError(
                    f"Invalid legacy success in {answers_path} at line "
                    f"{line_number}: a successful record must have a "
                    "non-empty cleaned_answer and cannot be resumed until "
                    "its status metadata is repaired."
                )
            if record["raw_answer"] is not None and not isinstance(
                record["raw_answer"], str
            ):
                raise ValueError(
                    f"Invalid answer record in {answers_path} at line "
                    f"{line_number}: raw_answer must be a string or null."
                )
            if record["error"] is not None and not isinstance(
                record["error"], str
            ):
                raise ValueError(
                    f"Invalid answer record in {answers_path} at line "
                    f"{line_number}: error must be a string or null."
                )
            elapsed = record["elapsed_seconds"]
            if (
                not isinstance(elapsed, (int, float))
                or isinstance(elapsed, bool)
                or elapsed < 0
            ):
                raise ValueError(
                    f"Invalid answer record in {answers_path} at line "
                    f"{line_number}: elapsed_seconds must be nonnegative."
                )

            record_count += 1
            conversation_id = record.get("conversation_id")
            if isinstance(conversation_id, str) and conversation_id:
                conversation_ids.add(conversation_id)
            if status == "success":
                successful_count += 1
                completed.add(
                    case_key(record.get("paper_id"), record.get("question_id"))
                )
            else:
                failed_count += 1

    return (
        completed,
        conversation_ids,
        record_count,
        successful_count,
        failed_count,
    )


def freeze_resume_generator_settings(
    manifest,
    answers_path,
    generator_settings,
):
    missing = object()
    if "generator_request" in manifest:
        recorded_settings = manifest["generator_request"]
    else:
        legacy_configuration = manifest.get("configuration")
        if legacy_configuration is not None and not isinstance(
            legacy_configuration, dict
        ):
            raise ValueError(
                "Resume manifest configuration must be an object."
            )
        recorded_settings = (
            legacy_configuration["generator_request"]
            if isinstance(legacy_configuration, dict)
            and "generator_request" in legacy_configuration
            else missing
        )

    if recorded_settings is missing:
        answers_path = Path(answers_path)
        if answers_path.exists():
            if not answers_path.is_file():
                raise ValueError(
                    f"Resume answers path is not a file: {answers_path}"
                )
            with answers_path.open("r", encoding="utf-8") as handle:
                has_legacy_records = any(line.strip() for line in handle)
        else:
            has_legacy_records = False

        if has_legacy_records:
            raise ValueError(
                "Cannot resume reproducibly: generation_manifest.json does "
                "not record configuration.generator_request, but answers.jsonl "
                "already contains legacy records. Start a new run or perform "
                "an explicit audited provenance repair."
            )
        manifest["generator_request"] = dict(generator_settings)
        return

    if recorded_settings != generator_settings:
        raise ValueError(
            "Resume generator request settings do not match the current "
            "frozen configuration."
        )


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def write_json(path, payload):
    atomic_write_json(path, payload)


class GenerationCasePacer:
    """Minimum start-to-start interval for sequential benchmark cases."""

    def __init__(self, interval_seconds, clock_fn=time.monotonic, sleep_fn=time.sleep):
        self.interval_seconds = interval_seconds
        self._clock = clock_fn
        self._sleep = sleep_fn
        self._last_started = None
        self.total_sleep_seconds = 0.0

    def begin_case(self):
        now = self._clock()
        wait = 0.0
        if self._last_started is not None:
            wait = max(0.0, self._last_started + self.interval_seconds - now)
        if wait > 0:
            self._sleep(wait)
            self.total_sleep_seconds += wait
        self._last_started = self._clock()
        return wait


def validate_kb_state(kb_state, paper_id):
    missing = []
    if not kb_state or not kb_state.get("documents"):
        missing.append("documents")
    if not kb_state or kb_state.get("vectordb") is None:
        missing.append("vectordb")
    if not kb_state or not kb_state.get("bm25_chunks"):
        missing.append("BM25 chunks")
    if not kb_state or kb_state.get("bm25_index") is None:
        missing.append("BM25 index")
    if not kb_state or kb_state.get("rerank_model") is None:
        missing.append("reranker model")
    if missing:
        raise RuntimeError(
            f"Invalid knowledge base for paper {paper_id}; missing: "
            f"{', '.join(missing)}."
        )


def extract_returned_answer(chat_result):
    if not isinstance(chat_result, (tuple, list)) or len(chat_result) != 5:
        raise RuntimeError("chat() returned an unexpected result shape.")
    history = chat_result[0]
    if not isinstance(history, list) or not history:
        raise RuntimeError("chat() returned no chat history.")
    final_message = history[-1]
    if (
        not isinstance(final_message, dict)
        or final_message.get("role") != "assistant"
        or not isinstance(final_message.get("content"), str)
    ):
        raise RuntimeError("chat() returned no assistant answer.")
    return final_message["content"]


def build_application_logs(
    global_log_path,
    output_path,
    start_offset,
    conversation_ids,
):
    matched = []
    malformed_lines = 0
    global_log_path = Path(global_log_path)

    if global_log_path.exists():
        current_size = global_log_path.stat().st_size
        if current_size < start_offset:
            raise RuntimeError(
                "The global application log is shorter than the recorded "
                "run offset; it may have been replaced or truncated."
            )

        with global_log_path.open("rb") as handle:
            handle.seek(start_offset)
            appended_bytes = handle.read()

        for raw_line in appended_bytes.splitlines():
            if not raw_line.strip():
                continue
            try:
                entry = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                malformed_lines += 1
                continue
            if entry.get("conversation_id") in conversation_ids:
                matched.append(entry)

    sanitized = [sanitize_application_log(entry) for entry in matched]
    content = "".join(
        json.dumps(entry, ensure_ascii=False) + "\n" for entry in sanitized
    )
    atomic_write_text(output_path, content)

    return len(matched), malformed_lines, aggregate_checker_diagnostics(matched)


def make_manifest(
    args,
    run_id,
    run_dir,
    records,
    grouped_cases,
    log_path,
    log_offset,
    model_names,
    profile,
    generator_settings,
    rewriter_settings,
    relevance_checker_settings,
    groundedness_checker_settings,
    selection_provenance=None,
):
    return {
        "run_id": run_id,
        "stage": "qasper_headless_answer_generation",
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "path_base": PATH_BASE,
        "paths": {
            "qa_data": serialize_artifact_path(profile.qa_path),
            "pdf_directory": serialize_artifact_path(profile.pdf_path),
            "run_directory": serialize_artifact_path(run_dir),
            "answers": serialize_artifact_path(run_dir / "answers.jsonl"),
            "application_logs": serialize_artifact_path(
                run_dir / "application_logs.jsonl"
            ),
            "global_application_log": serialize_artifact_path(log_path),
        },
        "models": model_names,
        "evaluation_profile": profile_provenance(profile),
        "case_selection": selection_provenance,
        "operational": {
            "condition": "single_document",
            "empty_history_per_question": True,
            "max_papers": args.max_papers,
            "max_questions": args.max_questions,
            "generation_case_interval_seconds": (
                profile.generation_case_interval_seconds
            ),
            "generation_case_pacing_note": (
                "Start-to-start pacing reduces TPM/request bursts but cannot "
                "increase or reset TPD allowance; the benchmark must begin "
                "with sufficient daily quota, and concurrent use of the "
                "shared openai/gpt-oss-20b checker model can still exhaust "
                "organization quota."
            ),
        },
        "generation_pacing": {
            "total_sleep_seconds": 0.0,
        },
        "checker_diagnostics": empty_checker_diagnostics(),
        "generator_request": generator_settings,
        "rewriter_request": rewriter_settings,
        "relevance_checker_request": relevance_checker_settings,
        "groundedness_checker_request": groundedness_checker_settings,
        "counts": {
            "dataset_papers": len(records),
            "dataset_questions": sum(
                len(paper["qas"]["question"]) for paper in records
            ),
            "selected_papers": len(grouped_cases),
            "selected_questions": sum(
                len(indices) for _, indices in grouped_cases
            ),
            "successful": 0,
            "failed": 0,
            "skipped_completed": 0,
            "answer_records": 0,
            "application_log_records": 0,
            "malformed_new_application_log_lines": 0,
        },
        "application_log_start_offset": log_offset,
    }


def validate_resume_manifest(
    manifest,
    args,
    run_id,
    profile,
    answers_path,
    selection_provenance=None,
):
    if manifest.get("run_id") != run_id:
        raise ValueError("Resume manifest run ID does not match --run-id.")
    configuration = manifest.get("operational")
    if configuration is None:
        configuration = manifest.get("configuration", {})
    expected = {
        "max_papers": args.max_papers,
        "max_questions": args.max_questions,
    }
    actual = {key: configuration.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            "Resume limits must match the original run: "
            f"expected {actual}, received {expected}."
        )
    if manifest.get("case_selection") != selection_provenance:
        raise ValueError(
            "Resume case-selection provenance does not match the original run."
        )
    if not isinstance(manifest.get("application_log_start_offset"), int):
        raise ValueError("Resume manifest has no valid application-log offset.")
    return validate_generation_profile_provenance(
        manifest,
        profile,
        records_path=answers_path,
        allow_legacy_read=False,
    )


def run_generation(args, clock_fn=time.monotonic, sleep_fn=time.sleep):
    if args.run_id is None:
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
    else:
        run_id = args.run_id
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "Run ID must start with an alphanumeric character and contain "
            "only letters, numbers, dots, underscores, or hyphens."
        )

    run_dir = RUNS_DIR / run_id
    answers_path = run_dir / "answers.jsonl"
    application_logs_path = run_dir / "application_logs.jsonl"
    manifest_path = run_dir / "generation_manifest.json"
    profile = load_evaluation_config()

    records = load_qasper_records(profile.qa_path)
    source_run_id = getattr(args, "select_below_threshold_from_run", None)
    source_judge_id = getattr(args, "selection_judge_id", None)
    if bool(source_run_id) != bool(source_judge_id):
        raise ValueError(
            "Failure-subset selection requires both source run and judge IDs."
        )
    if source_run_id and (
        args.max_papers is not None or args.max_questions is not None
    ):
        raise ValueError(
            "--max-papers and --max-questions cannot be combined with "
            "failure-subset selection."
        )
    selection = (
        load_below_threshold_selection(
            source_run_id,
            source_judge_id,
            profile.dataset_sha256,
            profile.qa_path,
        )
        if source_run_id
        else None
    )
    grouped_cases = select_cases(
        records,
        max_papers=args.max_papers,
        max_questions=args.max_questions,
        case_keys=selection.case_keys if selection else None,
    )
    if not grouped_cases:
        raise ValueError("The selected QASPER subset contains no questions.")

    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume; manifest not found: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        has_profile_provenance = validate_resume_manifest(
            manifest,
            args,
            run_id,
            profile,
            answers_path,
            selection.provenance if selection else None,
        )
    else:
        if run_dir.exists():
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. "
                "Use --resume to continue it."
            )
        run_dir.mkdir(parents=True)
        manifest = None
        has_profile_provenance = False

    from chat.service import chat
    from config import (
        EMBEDDING_MODEL_NAME,
        GENERATOR_MODEL,
        GROUNDEDNESS_CHECKER,
        RELEVANCE_CHECKER,
        RERANK_MODEL_NAME,
        REWRITER_MODEL,
    )
    from core.models import (
        generation_request_settings,
        groundedness_checker_request_settings,
        initialize_models,
        relevance_checker_request_settings,
        rewriter_request_settings,
    )
    from core.state import delete_vectorstore, new_kb_state
    from telemetry.logger import EVAL_LOG_PATH
    from ingestion.indexing import index_files

    model_names = {
        "embedding": EMBEDDING_MODEL_NAME,
        "reranker": RERANK_MODEL_NAME,
        "generator": GENERATOR_MODEL,
        "rewriter": REWRITER_MODEL,
        "relevance_checker": RELEVANCE_CHECKER,
        "groundedness_checker": GROUNDEDNESS_CHECKER,
    }
    generator_settings = generation_request_settings()
    rewriter_settings = rewriter_request_settings()
    relevance_checker_settings = relevance_checker_request_settings()
    groundedness_checker_settings = groundedness_checker_request_settings()

    if manifest is None:
        log_offset = (
            EVAL_LOG_PATH.stat().st_size
            if EVAL_LOG_PATH.exists()
            else 0
        )
        manifest = make_manifest(
            args,
            run_id,
            run_dir,
            records,
            grouped_cases,
            EVAL_LOG_PATH,
            log_offset,
            model_names,
            profile,
            generator_settings,
            rewriter_settings,
            relevance_checker_settings,
            groundedness_checker_settings,
            selection.provenance if selection else None,
        )
        write_json(manifest_path, manifest)
    else:
        log_offset = manifest["application_log_start_offset"]
        freeze_resume_generator_settings(
            manifest,
            answers_path,
            generator_settings,
        )
        recorded_rewriter_settings = manifest.get("rewriter_request")
        if (
            recorded_rewriter_settings is not None
            and recorded_rewriter_settings != rewriter_settings
        ):
            raise ValueError(
                "Resume rewriter request settings do not match the recorded "
                "runtime provenance."
            )
        recorded_relevance_settings = manifest.get(
            "relevance_checker_request"
        )
        recorded_groundedness_settings = manifest.get(
            "groundedness_checker_request"
        )
        legacy_checker_settings = manifest.get("checker_request")
        if recorded_relevance_settings is None:
            recorded_relevance_settings = legacy_checker_settings
        if recorded_groundedness_settings is None:
            recorded_groundedness_settings = legacy_checker_settings
        for role, recorded, current in (
            (
                "relevance",
                recorded_relevance_settings,
                relevance_checker_settings,
            ),
            (
                "groundedness",
                recorded_groundedness_settings,
                groundedness_checker_settings,
            ),
        ):
            if recorded is not None and recorded != current:
                raise ValueError(
                    f"Resume {role} checker request settings do not match "
                    "the recorded runtime provenance."
                )
        if not has_profile_provenance:
            manifest["path_base"] = PATH_BASE
            manifest["paths"] = make_manifest(
                args,
                run_id,
                run_dir,
                records,
                grouped_cases,
                EVAL_LOG_PATH,
                log_offset,
                model_names,
                profile,
                generator_settings,
                rewriter_settings,
                relevance_checker_settings,
                groundedness_checker_settings,
                selection.provenance if selection else None,
            )["paths"]
            manifest["evaluation_profile"] = profile_provenance(profile)
        manifest["status"] = "running"
        manifest["finished_at"] = None
        manifest["models"] = model_names

    (
        completed,
        conversation_ids,
        record_count,
        successful_count,
        failed_count,
    ) = load_existing_run_records(answers_path)
    manifest["counts"]["answer_records"] = record_count
    manifest["counts"]["successful"] = successful_count
    manifest["counts"]["failed"] = failed_count
    manifest["counts"]["skipped_completed"] = 0
    write_json(manifest_path, manifest)

    recorded_pacing_sleep = (
        (manifest.get("generation_pacing") or {}).get("total_sleep_seconds", 0.0)
    )
    if (
        not isinstance(recorded_pacing_sleep, (int, float))
        or isinstance(recorded_pacing_sleep, bool)
        or recorded_pacing_sleep < 0
    ):
        raise ValueError("Generation manifest has invalid pacing telemetry.")
    _, reranker = initialize_models()
    kb_state = new_kb_state(mode="single")
    pacer = GenerationCasePacer(
        profile.generation_case_interval_seconds,
        clock_fn=clock_fn,
        sleep_fn=sleep_fn,
    )

    try:
        for paper, question_indices in grouped_cases:
            paper_id = paper["id"]
            pending_indices = [
                index for index in question_indices
                if case_key(
                    paper_id,
                    paper["qas"]["question_id"][index],
                ) not in completed
            ]
            skipped = len(question_indices) - len(pending_indices)
            manifest["counts"]["skipped_completed"] += skipped
            if not pending_indices:
                continue

            pdf_id = paper_id.removeprefix("arXiv:")
            pdf_path = profile.pdf_path / f"{pdf_id}.pdf"
            indexing_error = None

            try:
                if not pdf_path.is_file():
                    raise FileNotFoundError(
                        f"QASPER PDF not found: {pdf_path}"
                    )
                kb_state, _, _, _ = index_files(
                    [str(pdf_path)],
                    kb_state,
                    "single",
                    profile.chunk_size,
                    profile.chunk_overlap,
                )
                kb_state["rerank_model"] = reranker
                validate_kb_state(kb_state, paper_id)
            except Exception as exc:
                indexing_error = (
                    f"{type(exc).__name__}: {sanitize_error(exc)}"
                )

            qas = paper["qas"]
            for question_index in pending_indices:
                pacer.begin_case()
                question_id = qas["question_id"][question_index]
                question = qas["question"][question_index]
                conversation_id = str(uuid.uuid4())
                conversation_ids.add(conversation_id)
                started = perf_counter()
                raw_answer = None
                cleaned_answer = None
                status = "error"
                error = indexing_error

                if indexing_error is None:
                    try:
                        chat_result = chat(
                            question,
                            [],
                            kb_state,
                            profile.retrieval_mode,
                            profile.retrieval_top_n,
                            profile.rerank_candidate_count,
                            profile.rerank_top_k,
                            profile.fusion_threshold,
                            False,
                            "all",
                            [],
                            conversation_id,
                        )
                        raw_answer = extract_returned_answer(chat_result)
                        cleaned_answer, status, error = classify_returned_answer(
                            raw_answer
                        )
                        if status == "success":
                            completed.add(case_key(paper_id, question_id))
                    except Exception as exc:
                        error = sanitize_error(exc)

                answer_record = {
                    "paper_id": paper_id,
                    "question_id": question_id,
                    "question": question,
                    "gold_answers": qas["answers"][question_index],
                    "conversation_id": conversation_id,
                    "raw_answer": raw_answer,
                    "cleaned_answer": cleaned_answer,
                    "status": status,
                    "error": error,
                    "elapsed_seconds": round(perf_counter() - started, 6),
                }
                append_jsonl(answers_path, answer_record)
                record_count += 1
                manifest["counts"]["answer_records"] = record_count
                manifest["counts"][
                    "successful" if status == "success" else "failed"
                ] += 1
                manifest["generation_pacing"]["total_sleep_seconds"] = round(
                    recorded_pacing_sleep + pacer.total_sleep_seconds,
                    6,
                )
                (
                    application_log_count,
                    malformed_log_lines,
                    checker_diagnostics,
                ) = build_application_logs(
                    EVAL_LOG_PATH,
                    application_logs_path,
                    log_offset,
                    conversation_ids,
                )
                manifest["counts"][
                    "application_log_records"
                ] = application_log_count
                manifest["counts"][
                    "malformed_new_application_log_lines"
                ] = malformed_log_lines
                manifest["checker_diagnostics"] = checker_diagnostics
                write_json(manifest_path, manifest)

        (
            application_log_count,
            malformed_log_lines,
            checker_diagnostics,
        ) = build_application_logs(
            EVAL_LOG_PATH,
            application_logs_path,
            log_offset,
            conversation_ids,
        )
        manifest["counts"][
            "application_log_records"
        ] = application_log_count
        manifest["counts"][
            "malformed_new_application_log_lines"
        ] = malformed_log_lines
        manifest["checker_diagnostics"] = checker_diagnostics
        selected_keys = {
            case_key(paper["id"], paper["qas"]["question_id"][index])
            for paper, indices in grouped_cases
            for index in indices
        }
        diagnostic_unknown = sum(
            values["unknown"] for values in checker_diagnostics.values()
        )
        manifest["status"] = (
            "complete"
            if selected_keys.issubset(completed)
            and diagnostic_unknown == 0
            and application_log_count >= manifest["counts"]["successful"]
            else "complete_with_errors"
        )
        manifest["finished_at"] = utc_now()
        write_json(manifest_path, manifest)
    except BaseException:
        manifest["status"] = "interrupted"
        manifest["finished_at"] = utc_now()
        write_json(manifest_path, manifest)
        raise
    finally:
        delete_vectorstore(
            kb_state.get("vectordb") if kb_state else None
        )

    print(
        f"Run {run_id} finished with status {manifest['status']}. "
        f"Answers: {answers_path}"
    )
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate headless single-document QASPER answers."
    )
    parser.add_argument(
        "--run-id",
        help="Run directory name; generated automatically when omitted.",
    )
    parser.add_argument(
        "--max-papers",
        type=positive_int,
        help="Use only the first N papers in the fixed saved dataset.",
    )
    parser.add_argument(
        "--max-questions",
        type=positive_int,
        help="Use only the first N questions total in dataset order.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run with the same ID and limits.",
    )
    parser.add_argument(
        "--select-below-threshold-from-run",
        help=(
            "Regenerate only cases below the recorded semantic threshold in "
            "this source run."
        ),
    )
    parser.add_argument(
        "--selection-judge-id",
        help="Source judge ID used for below-threshold case selection.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_generation(args)


if __name__ == "__main__":
    main()
