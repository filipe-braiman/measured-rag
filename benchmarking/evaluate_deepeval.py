"""Evaluate generated answers with DeepEval semantic correctness scoring."""

import argparse
import asyncio
import json
import math
import os
import random
import re
import time
import uuid
import copy
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Keep this standalone evaluator local and prevent DeepEval from loading provider credentials or sending anonymous telemetry during import/evaluation.
os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncGroq,
    Groq,
    RateLimitError,
)

from benchmarking.evaluation_config import (
    PATH_BASE,
    REPO_ROOT,
    LEGACY_RESOLVED_CONFIG_FILENAME,
    config_from_provenance,
    default_reasoning_effort,
    load_evaluation_config,
    load_legacy_evaluation_config,
    profile_provenance,
    sanitize_error as sanitize_error_text,
    serialize_artifact_path,
    validate_generation_profile_scope,
    validate_generation_profile_provenance,
    validate_profile_provenance,
)
from benchmarking.artifact_io import atomic_write_json
from benchmarking.evaluate_qasper_f1 import (
    load_manifest as load_generation_manifest,
    read_answer_attempts,
    validate_gold_bundle,
)
from benchmarking.qasper_judge_input import (
    ANNOTATION_EVIDENCE_POLICY,
    evidence_for_reference,
    format_expected_output as format_judge_expected_output,
    validate_serialized_evidence,
)


EVAL_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVAL_DIR / "runs"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
JUDGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
EVALUATOR_VERSION = "qasper-reference-geval-v1.6"
METRIC_NAME = "QASPER Reference-Based Answer Correctness"
SUPPORTED_MODEL_CONFIG = {
    "qwen/qwen3.6-27b": {"none", "default"},
    "qwen/qwen3.8-27b": {"none", "default"},
    "openai/gpt-oss-120b": {"low", "medium", "high"},
}
EVALUATION_STEPS = [
    "Identify the exact fact or conclusion requested by the question in INPUT.",
    "Treat the reference answer as acceptable, not as an exhaustive description of every correct response.",
    "Use only this annotation's supplied gold evidence to understand and verify its reference answer.",
    "Judge whether ACTUAL_OUTPUT communicates the same central fact or a compatible answer supported by that evidence.",
    "Accept correct paraphrases, explanations, and more specific answers.",
    "Absence of a detail from a short reference is not evidence that the detail is fabricated.",
    "Penalize an additional claim only when it materially changes the answer or is contradicted by the reference or supplied gold evidence.",
    "Do not require exact wording when the meaning is equivalent.",
    "For yes/no questions, require the correct conclusion.",
    (
        "For an Unanswerable reference, treat an explicit abstention as "
        "correct and reject a substantive answer that conflicts with the "
        "unanswerable annotation."
    ),
    (
        "Treat tokens matching BIBREF<n>, TABREF<n>, FIGREF<n>, or SECREF<n> "
        "as non-semantic QASPER annotation markup. Their numeric suffixes do "
        "not necessarily equal human-visible citation, table, figure, or "
        "section numbers. Do not require ACTUAL_OUTPUT to repeat them, and "
        "do not treat a natural label such as Table 6 as contradictory to "
        "TABREF19 solely because the numbers differ."
    ),
    (
        "Treat INLINEFORM<n> as a marker for omitted mathematical content. "
        "Do not require the literal marker and do not invent its missing "
        "mathematical value; judge only substantive information that remains "
        "available in the supplied reference and gold evidence."
    ),
    "Evaluate all ordinary factual content normally.",
    (
        "For a locator-style reference such as column Ens Test in Table "
        "TABREF19, do not presume that additional values are fabricated "
        "solely because the abbreviated reference does not reproduce them."
    ),
    (
        "Apply the fixed rubric to correctness of the central answer and "
        "give a concise reason."
    ),
]
QASPER_PLACEHOLDER_POLICY = {
    "policy_version": "qasper-placeholder-markup-v1",
    "non_semantic_cross_reference_patterns": [
        "BIBREF<n>",
        "TABREF<n>",
        "FIGREF<n>",
        "SECREF<n>",
    ],
    "omitted_mathematical_content_pattern": "INLINEFORM<n>",
    "reference_transformation": "none",
    "original_converted_reference_preserved": True,
}
RUBRIC_DEFINITION = [
    {
        "score_range": [0, 1],
        "expected_outcome": (
            "Wrong conclusion, incompatible answer, or fabricated substantive "
            "answer for an unanswerable case."
        ),
    },
    {
        "score_range": [2, 4],
        "expected_outcome": (
            "Mostly incorrect, seriously incomplete, or materially contradictory."
        ),
    },
    {
        "score_range": [5, 6],
        "expected_outcome": (
            "Partially correct but missing or misstating an important part."
        ),
    },
    {
        "score_range": [7, 8],
        "expected_outcome": (
            "Central answer is correct with a minor omission or non-material issue."
        ),
    },
    {
        "score_range": [9, 10],
        "expected_outcome": (
            "Central answer is fully correct; compatible explanation or "
            "additional detail is allowed."
        ),
    },
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def package_version(distribution):
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


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


def resolve_model_configuration(
    model_id,
    requested_effort=None,
    max_completion_tokens=None,
):
    if max_completion_tokens is None:
        max_completion_tokens = (
            load_evaluation_config().judge_max_completion_tokens
        )
    configuration = SUPPORTED_MODEL_CONFIG.get(model_id)
    if configuration is None:
        raise ValueError(
            f"Unsupported judge model {model_id!r}; supported models are: "
            + ", ".join(SUPPORTED_MODEL_CONFIG)
        )
    effort = requested_effort or default_reasoning_effort(model_id)
    if effort not in configuration:
        allowed = ", ".join(sorted(configuration))
        raise ValueError(
            f"Reasoning effort {effort!r} is not supported by {model_id}; "
            f"choose one of: {allowed}."
        )
    return {
        "model_id": model_id,
        "reasoning_effort": effort,
        "request_parameters": {
            "temperature": 0.0,
            "seed": 0,
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": effort,
            "reasoning_format": "hidden",
            "stream": False,
            "timeout": 120.0,
        },
    }


def sanitize_error(error):
    return sanitize_error_text(error)


TELEMETRY_COUNT_FIELDS = (
    "provider_attempt_count",
    "retry_count",
    "rate_limit_response_count",
    "rate_limit_retry_count",
    "other_transient_retry_count",
    "schema_output_retry_count",
    "retry_after_used_count",
)
TELEMETRY_SECONDS_FIELDS = (
    "pacing_sleep_seconds",
    "retry_backoff_sleep_seconds",
)


def empty_transport_telemetry():
    telemetry = {field: 0 for field in TELEMETRY_COUNT_FIELDS}
    telemetry.update({field: 0.0 for field in TELEMETRY_SECONDS_FIELDS})
    telemetry["final_error_categories"] = {}
    return telemetry


def transport_telemetry_delta(before, after):
    delta = {
        field: after[field] - before[field]
        for field in TELEMETRY_COUNT_FIELDS + TELEMETRY_SECONDS_FIELDS
    }
    categories = {}
    for category, count in after["final_error_categories"].items():
        difference = count - before["final_error_categories"].get(category, 0)
        if difference:
            categories[category] = difference
    delta["final_error_categories"] = categories
    return delta


def score_transport_telemetry(delta=None, final_error_category=None):
    delta = delta or empty_transport_telemetry()
    return {
        "provider_attempt_count": int(delta["provider_attempt_count"]),
        "retry_count": int(delta["retry_count"]),
        "rate_limit_response_count": int(delta["rate_limit_response_count"]),
        "rate_limit_retry_count": int(delta["rate_limit_retry_count"]),
        "other_transient_retry_count": int(
            delta["other_transient_retry_count"]
        ),
        "schema_output_retry_count": int(delta["schema_output_retry_count"]),
        "pacing_sleep_seconds": round(delta["pacing_sleep_seconds"], 6),
        "retry_backoff_sleep_seconds": round(
            delta["retry_backoff_sleep_seconds"], 6
        ),
        "retry_after_used": bool(delta["retry_after_used_count"]),
        "final_provider_error_category": final_error_category,
        "final_http_status": None,
        "final_provider_error_type": None,
        "final_provider_error_code": None,
        "final_retry_decision": None,
    }


def _status_code(error):
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def _normalized_provider_token(value):
    if not isinstance(value, str) or not value.strip():
        return None
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")[:100]


def provider_error_metadata(error):
    """Extract only safe structured SDK metadata, never body text."""
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return {"status": _status_code(error), "type": None, "code": None}
    detail = body.get("error", body)
    if not isinstance(detail, dict):
        detail = {}
    return {
        "status": _status_code(error),
        "type": _normalized_provider_token(detail.get("type")),
        "code": _normalized_provider_token(detail.get("code")),
    }


SCHEMA_PROVIDER_CODES = {
    "failed_generation",
    "invalid_json",
    "json_generation_failed",
    "json_schema_validation_failed",
    "json_validate_failed",
    "malformed_json",
    "model_output_invalid",
    "schema_generation_failed",
    "schema_output_validation_failed",
}


def classify_request_error(error, *, output_phase=False):
    """Return the stable category and whether the failure may be retried."""
    if output_phase:
        return "schema_output", True
    status = _status_code(error)
    if isinstance(error, RateLimitError) or status == 429:
        return "rate_limit", True
    if isinstance(error, (APITimeoutError, TimeoutError, asyncio.TimeoutError)):
        return "timeout", True
    if isinstance(error, (APIConnectionError, ConnectionError)):
        return "connection", True
    if isinstance(error, APIStatusError) or status is not None:
        if status is not None and 500 <= status <= 599:
            return "transient_http", True
        metadata = provider_error_metadata(error)
        structured_tokens = {metadata["type"], metadata["code"]} - {None}
        if status in {400, 422} and structured_tokens & SCHEMA_PROVIDER_CODES:
            return "schema_output", True
        return "permanent_http", False
    return "permanent_error", False


def _retry_after_seconds(error):
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is None:
        return None
    value = None
    try:
        value = headers.get("retry-after")
    except AttributeError:
        pass
    if value is None:
        try:
            for key, candidate in headers.items():
                if str(key).lower() == "retry-after":
                    value = candidate
                    break
        except (AttributeError, TypeError):
            return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def stable_schema_name(schema):
    raw = getattr(schema, "__name__", None) or schema.__class__.__name__
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw)).strip("_")
    return (normalized or "deepeval_response")[:64]


def strict_provider_schema(schema_definition):
    """Return and validate Groq's strict JSON-Schema subset."""
    if not isinstance(schema_definition, dict):
        raise TypeError("DeepEval schema must be a JSON object.")
    converted = copy.deepcopy(schema_definition)

    def convert(node, location="$"):
        if isinstance(node, list):
            for index, item in enumerate(node):
                convert(item, f"{location}[{index}]")
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" or "properties" in node:
            properties = node.get("properties")
            if not isinstance(properties, dict):
                raise ValueError(
                    f"Strict schema object at {location} has no properties object."
                )
            node["additionalProperties"] = False
            node["required"] = list(properties)
        for key, value in node.items():
            convert(value, f"{location}.{key}")

    convert(converted)
    if converted.get("type") != "object" and "properties" not in converted:
        raise ValueError("Strict provider schema root must be an object.")

    def validate(node, location="$"):
        if isinstance(node, list):
            for index, item in enumerate(node):
                validate(item, f"{location}[{index}]")
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" or "properties" in node:
            properties = node.get("properties")
            if node.get("additionalProperties") is not False:
                raise ValueError(
                    f"Strict schema object at {location} permits extra properties."
                )
            if node.get("required") != list(properties):
                raise ValueError(
                    f"Strict schema object at {location} does not require every property."
                )
        for key, value in node.items():
            validate(value, f"{location}.{key}")

    validate(converted)
    return converted


class JudgeRequestError(RuntimeError):
    """Sanitized terminal provider/output failure with a stable category."""

    def __init__(
        self,
        category,
        status_code=None,
        provider_error_type=None,
        provider_error_code=None,
        retry_decision="not_retryable",
    ):
        self.category = category
        self.status_code = status_code
        self.provider_error_type = provider_error_type
        self.provider_error_code = provider_error_code
        self.retry_decision = retry_decision
        suffix = f", HTTP {status_code}" if status_code is not None else ""
        super().__init__(f"Judge request failed ({category}{suffix}).")


class GroqJudgeModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model_configuration,
        max_retries=3,
        request_interval_seconds=6.0,
        retry_base_delay_seconds=2.0,
        retry_max_delay_seconds=60.0,
        retry_jitter_seconds=0.5,
        max_retry_after_seconds=300.0,
        client=None,
        async_client=None,
        sleep_fn=time.sleep,
        async_sleep_fn=asyncio.sleep,
        clock_fn=time.monotonic,
        random_fn=random.random,
    ):
        self.model_id = model_configuration["model_id"]
        self.reasoning_effort = model_configuration["reasoning_effort"]
        self.request_parameters = dict(
            model_configuration["request_parameters"]
        )
        self.max_retries = max_retries
        self.request_interval_seconds = request_interval_seconds
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.retry_max_delay_seconds = retry_max_delay_seconds
        self.retry_jitter_seconds = retry_jitter_seconds
        self.max_retry_after_seconds = max_retry_after_seconds
        self._sleep = sleep_fn
        self._async_sleep = async_sleep_fn
        self._clock = clock_fn
        self._random = random_fn
        self._last_attempt_started = None
        self._telemetry = empty_transport_telemetry()

        api_key = os.getenv("GROQ_API_KEY")
        if client is None and not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        self._client = client or Groq(api_key=api_key)
        self._async_client = async_client or AsyncGroq(api_key=api_key)
        super().__init__(model=self.model_id)

    def load_model(self, *args, **kwargs):
        return self._client

    def get_model_name(self, *args, **kwargs):
        return f"Groq:{self.model_id}"

    def supports_log_probs(self):
        return False

    def supports_temperature(self):
        return True

    def supports_structured_outputs(self):
        return True

    def supports_json_mode(self):
        return True

    def _request_arguments(self, prompt, schema=None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Judge prompt must be a non-empty string.")

        content = prompt
        arguments = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            **self.request_parameters,
        }
        if schema is not None:
            if hasattr(schema, "model_json_schema"):
                schema_definition = schema.model_json_schema()
            elif hasattr(schema, "schema"):
                schema_definition = schema.schema()
            else:
                raise TypeError("DeepEval supplied an unsupported schema type.")
            if self.model_id == "qwen/qwen3.8-27b":
                strict_schema = strict_provider_schema(schema_definition)
                schema_name = stable_schema_name(schema)
                arguments["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": strict_schema,
                    },
                }
            else:
                schema_text = json.dumps(schema_definition, ensure_ascii=False)
                arguments["messages"] = [
                    {
                        "role": "user",
                        "content": (
                            f"{prompt}\n\nReturn only one JSON object matching "
                            f"this schema:\n{schema_text}"
                        ),
                    }
                ]
                arguments["response_format"] = {"type": "json_object"}
        return arguments

    def telemetry_snapshot(self):
        return {
            **{
                field: self._telemetry[field]
                for field in TELEMETRY_COUNT_FIELDS + TELEMETRY_SECONDS_FIELDS
            },
            "final_error_categories": dict(
                self._telemetry["final_error_categories"]
            ),
        }

    def _bounded_jitter(self):
        value = self._random()
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            value = 0.0
        return min(max(float(value), 0.0), 1.0) * self.retry_jitter_seconds

    def _wait_requirements(self, retry_deadline):
        now = self._clock()
        pacing_wait = 0.0
        if self._last_attempt_started is not None:
            pacing_wait = max(
                0.0,
                self._last_attempt_started
                + self.request_interval_seconds
                - now,
            )
        retry_wait = (
            max(0.0, retry_deadline - now)
            if retry_deadline is not None
            else 0.0
        )
        wait = max(pacing_wait, retry_wait)
        pacing_component = min(wait, pacing_wait)
        retry_component = wait - pacing_component
        return wait, pacing_component, retry_component

    def _record_wait(self, pacing_seconds, retry_seconds):
        self._telemetry["pacing_sleep_seconds"] += pacing_seconds
        self._telemetry["retry_backoff_sleep_seconds"] += retry_seconds

    def _begin_attempt_sync(self, retry_deadline):
        wait, pacing_seconds, retry_seconds = self._wait_requirements(
            retry_deadline
        )
        if wait > 0:
            self._sleep(wait)
            self._record_wait(pacing_seconds, retry_seconds)
        self._last_attempt_started = self._clock()
        self._telemetry["provider_attempt_count"] += 1

    async def _begin_attempt_async(self, retry_deadline):
        wait, pacing_seconds, retry_seconds = self._wait_requirements(
            retry_deadline
        )
        if wait > 0:
            await self._async_sleep(wait)
            self._record_wait(pacing_seconds, retry_seconds)
        self._last_attempt_started = self._clock()
        self._telemetry["provider_attempt_count"] += 1

    def _terminal_error(self, category, error):
        categories = self._telemetry["final_error_categories"]
        categories[category] = categories.get(category, 0) + 1
        metadata = provider_error_metadata(error)
        raise JudgeRequestError(
            category,
            metadata["status"],
            metadata["type"],
            metadata["code"],
            "retries_exhausted" if category in {
                "rate_limit", "timeout", "connection", "transient_http",
                "schema_output",
            } else "not_retryable",
        ) from error

    def _retry_deadline(self, error, category, retry_index):
        if category == "rate_limit":
            self._telemetry["rate_limit_response_count"] += 1
            retry_after = _retry_after_seconds(error)
            if (
                retry_after is not None
                and retry_after > self.max_retry_after_seconds
            ):
                self._terminal_error("rate_limit_retry_after_exceeded", error)
            if retry_after is not None:
                self._telemetry["retry_after_used_count"] += 1
                delay = retry_after + self._bounded_jitter()
            else:
                delay = min(
                    self.retry_base_delay_seconds * (2**retry_index),
                    self.retry_max_delay_seconds,
                ) + self._bounded_jitter()
            self._telemetry["rate_limit_retry_count"] += 1
        else:
            delay = min(
                self.retry_base_delay_seconds * (2**retry_index),
                self.retry_max_delay_seconds,
            ) + self._bounded_jitter()
            if category == "schema_output":
                self._telemetry["schema_output_retry_count"] += 1
            else:
                self._telemetry["other_transient_retry_count"] += 1
        self._telemetry["retry_count"] += 1
        return self._clock() + delay

    def _handle_failure(self, error, category, retryable, retry_index):
        if not retryable or retry_index >= self.max_retries:
            if category == "rate_limit":
                self._telemetry["rate_limit_response_count"] += 1
            self._terminal_error(category, error)
        return self._retry_deadline(error, category, retry_index)

    def _prepare_arguments(self, prompt, schema):
        try:
            return self._request_arguments(prompt, schema=schema)
        except Exception as error:
            self._terminal_error("permanent_configuration", error)

    @staticmethod
    def _extract_content(response):
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Groq judge returned an empty response.")
        return content.strip()

    @staticmethod
    def _apply_schema(content, schema):
        if schema is None:
            return content
        if hasattr(schema, "model_validate_json"):
            return schema.model_validate_json(content)
        if hasattr(schema, "parse_raw"):
            return schema.parse_raw(content)
        raise TypeError("DeepEval supplied an unsupported schema type.")

    def generate(self, prompt, schema=None, **kwargs):
        arguments = self._prepare_arguments(prompt, schema)
        retry_deadline = None
        for retry_index in range(self.max_retries + 1):
            self._begin_attempt_sync(retry_deadline)
            try:
                response = self._client.chat.completions.create(**arguments)
            except Exception as error:
                category, retryable = classify_request_error(error)
                retry_deadline = self._handle_failure(
                    error, category, retryable, retry_index
                )
                continue
            try:
                content = self._extract_content(response)
                return self._apply_schema(content, schema)
            except Exception as error:
                retry_deadline = self._handle_failure(
                    error, "schema_output", True, retry_index
                )
        raise RuntimeError("Groq judge retry loop ended unexpectedly.")

    async def a_generate(self, prompt, schema=None, **kwargs):
        arguments = self._prepare_arguments(prompt, schema)
        retry_deadline = None
        for retry_index in range(self.max_retries + 1):
            await self._begin_attempt_async(retry_deadline)
            try:
                response = await self._async_client.chat.completions.create(
                    **arguments
                )
            except Exception as error:
                category, retryable = classify_request_error(error)
                retry_deadline = self._handle_failure(
                    error, category, retryable, retry_index
                )
                continue
            try:
                content = self._extract_content(response)
                return self._apply_schema(content, schema)
            except Exception as error:
                retry_deadline = self._handle_failure(
                    error, "schema_output", True, retry_index
                )
        raise RuntimeError("Groq judge retry loop ended unexpectedly.")


def build_rubric():
    return [
        Rubric(
            score_range=tuple(item["score_range"]),
            expected_outcome=item["expected_outcome"],
        )
        for item in RUBRIC_DEFINITION
    ]


def build_metric(model, threshold):
    return GEval(
        name=METRIC_NAME,
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=EVALUATION_STEPS,
        rubric=build_rubric(),
        model=model,
        threshold=threshold,
        async_mode=False,
        strict_mode=False,
        verbose_mode=False,
    )


def format_expected_output(reference, evidence):
    """Format one converted annotation and only its evidence for judging."""
    return format_judge_expected_output(reference, evidence)


def resume_key(paper_id, question_id, judge_model, evaluator_version):
    return json.dumps(
        [paper_id, question_id, judge_model, evaluator_version],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def prepare_cases(run_id, max_questions=None, profile=None):
    run_dir = RUNS_DIR / run_id
    generation_manifest_path = run_dir / "generation_manifest.json"
    answers_path = run_dir / "answers.jsonl"
    generation_manifest, selected_question_count = load_generation_manifest(
        generation_manifest_path,
        run_id,
    )
    if profile is not None:
        validate_generation_profile_provenance(
            generation_manifest,
            profile,
            records_path=answers_path,
            allow_legacy_read=False,
            allow_legacy_missing_case_pacing=True,
        )
    records, duplicate_attempt_count = read_answer_attempts(answers_path)
    if len(records) != selected_question_count:
        raise ValueError(
            "Generation manifest selected-question count does not match "
            f"unique answer cases: {selected_question_count} != {len(records)}."
        )
    if max_questions is not None:
        records = records[:max_questions]
    return run_dir, answers_path, records, duplicate_attempt_count


def prepare_case(record, judge_model, judge_id, threshold, attempt_number):
    case_label = f"{record['paper_id']}/{record['question_id']}"
    references = validate_gold_bundle(record["gold_answers"], case_label)
    return {
        "paper_id": record["paper_id"],
        "question_id": record["question_id"],
        "question": record["question"],
        "cleaned_actual_answer": record["cleaned_answer"],
        "references": references,
        "reference_evaluations": [],
        "generation_status": record["status"],
        "generation_error": record["error"],
        "judge_model": judge_model,
        "judge_id": judge_id,
        "deepeval_metric_name": METRIC_NAME,
        "threshold": threshold,
        "semantic_score": None,
        "passed": None,
        "judge_reason": None,
        "matched_reference_index": None,
        "matched_annotation_id": None,
        "matched_reference": None,
        "matched_answer_type": None,
        "status": None,
        "judge_error": None,
        "judge_transport": score_transport_telemetry(),
        "attempt_number": attempt_number,
        "elapsed_seconds": None,
        "evaluation_timestamp": None,
        "evaluator_version": EVALUATOR_VERSION,
        "resume_key": resume_key(
            record["paper_id"],
            record["question_id"],
            judge_model,
            EVALUATOR_VERSION,
        ),
    }


def aggregate_reference_transport(reference_evaluations):
    """Combine independently attributed reference telemetry at case level."""
    combined = score_transport_telemetry()
    for evaluation in reference_evaluations:
        telemetry = evaluation["judge_transport"]
        for field in (
            "provider_attempt_count",
            "retry_count",
            "rate_limit_response_count",
            "rate_limit_retry_count",
            "other_transient_retry_count",
            "schema_output_retry_count",
        ):
            combined[field] += telemetry[field]
        for field in (
            "pacing_sleep_seconds",
            "retry_backoff_sleep_seconds",
        ):
            combined[field] = round(combined[field] + telemetry[field], 6)
        combined["retry_after_used"] = (
            combined["retry_after_used"] or telemetry["retry_after_used"]
        )
        if telemetry.get("final_provider_error_category") is not None:
            for field in (
                "final_provider_error_category",
                "final_http_status",
                "final_provider_error_type",
                "final_provider_error_code",
                "final_retry_decision",
            ):
                combined[field] = telemetry.get(field)
    return combined


def evaluate_reference(
    record,
    reference,
    evidence,
    metric,
    threshold,
    telemetry_source=None,
):
    """Evaluate one prediction against one annotation and attribute telemetry."""
    telemetry_before = (
        telemetry_source.telemetry_snapshot()
        if telemetry_source is not None
        else None
    )
    terminal_request_error = None
    result = {
        "reference_index": reference["annotation_index"],
        "annotation_id": reference["annotation_id"],
        "answer_type": reference["answer_type"],
        "reference_answer": reference["answer"],
        "judge_evidence": evidence,
        "status": None,
        "semantic_score": None,
        "passed": None,
        "judge_reason": None,
        "judge_error": None,
        "judge_transport": score_transport_telemetry(),
    }
    try:
        test_case = LLMTestCase(
            input=record["question"],
            actual_output=record["cleaned_answer"],
            expected_output=format_expected_output(reference, evidence),
        )
        semantic_score = float(metric.measure(test_case, _show_indicator=False))
        if not 0.0 <= semantic_score <= 1.0:
            raise ValueError(
                f"DeepEval returned out-of-range score {semantic_score}."
            )
        result.update(
            {
                "status": "success",
                "semantic_score": semantic_score,
                "passed": semantic_score >= threshold,
                "judge_reason": metric.reason,
            }
        )
    except Exception as error:
        candidate = error
        while candidate is not None:
            if isinstance(candidate, JudgeRequestError):
                terminal_request_error = candidate
                break
            candidate = getattr(candidate, "__cause__", None)
        result.update(
            {
                "status": "judge_error",
                "judge_error": {
                    "type": type(error).__name__,
                    "message": sanitize_error(error),
                },
            }
        )

    if telemetry_source is not None:
        telemetry_after = telemetry_source.telemetry_snapshot()
        telemetry_delta = transport_telemetry_delta(
            telemetry_before,
            telemetry_after,
        )
        final_categories = telemetry_delta.pop("final_error_categories")
        final_category = sorted(final_categories)[0] if final_categories else None
        result["judge_transport"] = score_transport_telemetry(
            telemetry_delta,
            final_error_category=final_category,
        )
        if terminal_request_error is not None:
            result["judge_transport"].update(
                {
                    "final_http_status": terminal_request_error.status_code,
                    "final_provider_error_type": (
                        terminal_request_error.provider_error_type
                    ),
                    "final_provider_error_code": (
                        terminal_request_error.provider_error_code
                    ),
                    "final_retry_decision": terminal_request_error.retry_decision,
                }
            )
    return result


def evaluate_case(
    record,
    metric,
    judge_model,
    judge_id,
    threshold,
    attempt_number,
    telemetry_source=None,
):
    score_record = prepare_case(
        record,
        judge_model,
        judge_id,
        threshold,
        attempt_number,
    )
    started = time.perf_counter()
    if record["status"] != "success":
        score_record.update(
            {
                "semantic_score": 0.0,
                "passed": False,
                "judge_reason": (
                    "Generation failed, so semantic correctness is zero "
                    "without invoking the judge."
                ),
                "status": "generation_error",
            }
        )
    else:
        reference_evaluations = [
            evaluate_reference(
                record,
                reference,
                evidence_for_reference(record["gold_answers"], reference),
                metric,
                threshold,
                telemetry_source=telemetry_source,
            )
            for reference in score_record["references"]
        ]
        score_record["reference_evaluations"] = reference_evaluations
        score_record["judge_transport"] = aggregate_reference_transport(
            reference_evaluations
        )
        failures = [
            result
            for result in reference_evaluations
            if result["status"] != "success"
        ]
        if failures:
            first_failure = failures[0]
            score_record.update(
                {
                    "status": "judge_error",
                    "judge_error": first_failure["judge_error"],
                }
            )
        else:
            winner = max(
                reference_evaluations,
                key=lambda result: result["semantic_score"],
            )
            score_record.update(
                {
                    "semantic_score": winner["semantic_score"],
                    "passed": winner["semantic_score"] >= threshold,
                    "judge_reason": winner["judge_reason"],
                    "matched_reference_index": winner["reference_index"],
                    "matched_annotation_id": winner["annotation_id"],
                    "matched_reference": winner["reference_answer"],
                    "matched_answer_type": winner["answer_type"],
                    "status": "success",
                }
            )

    score_record["elapsed_seconds"] = round(
        time.perf_counter() - started,
        6,
    )
    score_record["evaluation_timestamp"] = utc_now()
    return score_record


def append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def write_json(path, payload):
    atomic_write_json(path, payload)


def validate_score_transport(record, label="score record"):
    telemetry = record.get("judge_transport")
    if not isinstance(telemetry, dict):
        raise ValueError(f"{label} is missing judge transport telemetry.")
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
            raise ValueError(f"{label} has invalid {field}.")
    for field in (
        "pacing_sleep_seconds",
        "retry_backoff_sleep_seconds",
    ):
        value = telemetry.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{label} has invalid {field}.")
    if not isinstance(telemetry.get("retry_after_used"), bool):
        raise ValueError(f"{label} has invalid retry_after_used.")
    category = telemetry.get("final_provider_error_category")
    if category is not None and (
        not isinstance(category, str) or not category.strip()
    ):
        raise ValueError(f"{label} has invalid final provider error category.")
    status = telemetry.get("final_http_status")
    if status is not None and (
        not isinstance(status, int) or isinstance(status, bool) or status < 100
    ):
        raise ValueError(f"{label} has invalid final HTTP status.")
    for field in ("final_provider_error_type", "final_provider_error_code"):
        value = telemetry.get(field)
        if value is not None and (
            not isinstance(value, str)
            or not value
            or not re.fullmatch(r"[a-z0-9_]+", value)
        ):
            raise ValueError(f"{label} has invalid {field}.")
    retry_decision = telemetry.get("final_retry_decision")
    if retry_decision not in {None, "not_retryable", "retries_exhausted"}:
        raise ValueError(f"{label} has invalid final retry decision.")
    if telemetry["retry_count"] > telemetry["provider_attempt_count"]:
        raise ValueError(f"{label} retry count exceeds provider attempts.")
    if telemetry["rate_limit_retry_count"] > telemetry["rate_limit_response_count"]:
        raise ValueError(f"{label} rate-limit retries exceed responses.")
    if telemetry["retry_after_used"] and telemetry["rate_limit_retry_count"] == 0:
        raise ValueError(f"{label} used retry-after without a rate-limit retry.")
    classified_retries = (
        telemetry["rate_limit_retry_count"]
        + telemetry["other_transient_retry_count"]
        + telemetry["schema_output_retry_count"]
    )
    if classified_retries != telemetry["retry_count"]:
        raise ValueError(f"{label} retry categories do not reconcile.")
    if record.get("status") == "generation_error" and any(
        telemetry[field] != 0 for field in count_fields
    ):
        raise ValueError(
            f"{label} generation error must have zero provider attempts."
        )
    if record.get("status") == "generation_error" and (
        telemetry["pacing_sleep_seconds"] != 0
        or telemetry["retry_backoff_sleep_seconds"] != 0
        or telemetry["retry_after_used"]
        or category is not None
    ):
        raise ValueError(f"{label} generation error has nonzero judge telemetry.")
    return telemetry


def validate_reference_evaluations(record, label="score record"):
    """Validate v1.4 reference coverage, max selection, and telemetry sums."""
    references = record.get("references")
    evaluations = record.get("reference_evaluations")
    if not isinstance(references, list) or not references:
        raise ValueError(f"{label} has no converted references.")
    if not isinstance(evaluations, list):
        raise ValueError(f"{label} has invalid reference evaluations.")

    if record.get("status") == "generation_error":
        if evaluations:
            raise ValueError(
                f"{label} generation error must not contain reference evaluations."
            )
        if record.get("semantic_score") != 0.0 or record.get("passed") is not False:
            raise ValueError(f"{label} has invalid generation-error scoring.")
        return {
            "reference_count": len(references),
            "successful_reference_count": 0,
            "reference_error_count": 0,
            "winner_non_first": False,
            "matched_answer_type": None,
        }

    if len(evaluations) != len(references):
        raise ValueError(
            f"{label} must contain one evaluation per converted reference."
        )
    successful = []
    failed = []
    for index, (reference, evaluation) in enumerate(zip(references, evaluations)):
        expected_identity = {
            "reference_index": reference.get("annotation_index"),
            "annotation_id": reference.get("annotation_id"),
            "answer_type": reference.get("answer_type"),
            "reference_answer": reference.get("answer"),
        }
        actual_identity = {
            field: evaluation.get(field) for field in expected_identity
        }
        if actual_identity != expected_identity or evaluation.get(
            "reference_index"
        ) != index:
            raise ValueError(f"{label} reference evaluation {index} is misaligned.")
        try:
            validate_serialized_evidence(evaluation.get("judge_evidence"))
        except ValueError as error:
            raise ValueError(
                f"{label} reference evaluation {index} has invalid evidence."
            ) from error
        status = evaluation.get("status")
        if status not in {"success", "judge_error"}:
            raise ValueError(
                f"{label} reference evaluation {index} has invalid status."
            )
        validate_score_transport(
            evaluation,
            f"{label} reference evaluation {index}",
        )
        if status == "success":
            score = evaluation.get("semantic_score")
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
                or not 0.0 <= score <= 1.0
            ):
                raise ValueError(
                    f"{label} reference evaluation {index} has invalid score."
                )
            if evaluation.get("passed") is not (score >= record["threshold"]):
                raise ValueError(
                    f"{label} reference evaluation {index} has invalid pass value."
                )
            if not isinstance(evaluation.get("judge_reason"), str):
                raise ValueError(
                    f"{label} reference evaluation {index} lacks a reason."
                )
            if evaluation.get("judge_error") is not None:
                raise ValueError(
                    f"{label} successful reference evaluation has an error."
                )
            successful.append(evaluation)
        else:
            if (
                evaluation.get("semantic_score") is not None
                or evaluation.get("passed") is not None
                or evaluation.get("judge_reason") is not None
                or not isinstance(evaluation.get("judge_error"), dict)
            ):
                raise ValueError(
                    f"{label} failed reference evaluation has invalid nullability."
                )
            failed.append(evaluation)

    expected_transport = aggregate_reference_transport(evaluations)
    actual_transport = validate_score_transport(record, label)
    for field, expected in expected_transport.items():
        actual = actual_transport.get(field)
        if isinstance(expected, float):
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"{label} case telemetry does not reconcile.")
        elif actual != expected:
            raise ValueError(f"{label} case telemetry does not reconcile.")

    matched_fields = (
        "matched_reference_index",
        "matched_annotation_id",
        "matched_reference",
        "matched_answer_type",
    )
    if failed:
        if record.get("status") != "judge_error":
            raise ValueError(f"{label} incomplete references require judge_error.")
        if record.get("semantic_score") is not None or record.get("passed") is not None:
            raise ValueError(f"{label} incomplete references require null result.")
        if any(record.get(field) is not None for field in matched_fields):
            raise ValueError(f"{label} incomplete references cannot select a winner.")
        winner = None
    else:
        if record.get("status") != "success":
            raise ValueError(f"{label} complete references require success.")
        winner = max(successful, key=lambda item: item["semantic_score"])
        expected_matched = {
            "matched_reference_index": winner["reference_index"],
            "matched_annotation_id": winner["annotation_id"],
            "matched_reference": winner["reference_answer"],
            "matched_answer_type": winner["answer_type"],
        }
        if any(record.get(field) != value for field, value in expected_matched.items()):
            raise ValueError(f"{label} matched-reference selection is invalid.")
        if record.get("semantic_score") != winner["semantic_score"]:
            raise ValueError(f"{label} does not retain the maximum reference score.")
        if record.get("passed") is not (winner["semantic_score"] >= record["threshold"]):
            raise ValueError(f"{label} has invalid case pass value.")
        if record.get("judge_reason") != winner["judge_reason"]:
            raise ValueError(f"{label} reason does not come from the winner.")

    return {
        "reference_count": len(references),
        "successful_reference_count": len(successful),
        "reference_error_count": len(failed),
        "winner_non_first": bool(winner and winner["reference_index"] != 0),
        "matched_answer_type": winner["answer_type"] if winner else None,
    }


def load_score_attempts(path, judge_model):
    attempts = {}
    if not path.exists():
        return attempts
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed JSON in {path} at line {line_number}."
                ) from error
            required = {
                "paper_id",
                "question_id",
                "judge_model",
                "evaluator_version",
                "resume_key",
                "status",
                "attempt_number",
                "judge_transport",
            }
            missing = required - record.keys()
            if missing:
                raise ValueError(
                    f"Score line {line_number} is missing fields: "
                    + ", ".join(sorted(missing))
                )
            expected_key = resume_key(
                record["paper_id"],
                record["question_id"],
                record["judge_model"],
                record["evaluator_version"],
            )
            if record["resume_key"] != expected_key:
                raise ValueError(
                    f"Score line {line_number} has an invalid resume key."
                )
            if record["judge_model"] != judge_model:
                raise ValueError(
                    f"Score line {line_number} belongs to a different judge model."
                )
            if record["evaluator_version"] != EVALUATOR_VERSION:
                raise ValueError(
                    f"Score line {line_number} uses an incompatible evaluator version."
                )
            if record["status"] not in {
                "success",
                "generation_error",
                "judge_error",
            }:
                raise ValueError(
                    f"Score line {line_number} has an invalid status."
                )
            if (
                not isinstance(record["attempt_number"], int)
                or isinstance(record["attempt_number"], bool)
                or record["attempt_number"] < 1
            ):
                raise ValueError(
                    f"Score line {line_number} has an invalid attempt number."
                )
            validate_score_transport(record, f"Score line {line_number}")
            validate_reference_evaluations(record, f"Score line {line_number}")
            attempts.setdefault(expected_key, []).append(record)
    return attempts


def select_latest_scores(attempts):
    selected = {}
    for key, records in attempts.items():
        successful = [record for record in records if record["status"] == "success"]
        generation_errors = [
            record for record in records if record["status"] == "generation_error"
        ]
        if successful:
            selected[key] = successful[-1]
        elif generation_errors:
            selected[key] = generation_errors[-1]
        else:
            selected[key] = records[-1]
    return selected


def validate_attempt_annotation_evidence(attempts, selected_cases, judge_model):
    """Bind every resumed reference judgment to its generation annotation."""
    cases_by_key = {
        resume_key(
            record["paper_id"],
            record["question_id"],
            judge_model,
            EVALUATOR_VERSION,
        ): record
        for record in selected_cases
    }
    if not set(attempts).issubset(cases_by_key):
        raise ValueError("Score history contains a case outside this judge selection.")
    for key, records in attempts.items():
        generation_record = cases_by_key[key]
        expected_references = validate_gold_bundle(
            generation_record["gold_answers"],
            f"{generation_record['paper_id']}/{generation_record['question_id']}",
        )
        for score in records:
            if score.get("references") != expected_references:
                raise ValueError(
                    "Score history contains references that do not match "
                    "generation provenance."
                )
            for reference, evaluation in zip(
                score["references"], score["reference_evaluations"]
            ):
                expected = evidence_for_reference(
                    generation_record["gold_answers"], reference
                )
                if evaluation.get("judge_evidence") != expected:
                    raise ValueError(
                        "Score history contains annotation evidence that does "
                        "not match generation provenance."
                    )


def reliability_aggregates(selected_scores):
    records = list(selected_scores.values())
    covered = []
    for record in records:
        telemetry = record.get("judge_transport")
        if telemetry is not None:
            covered.append(validate_score_transport(record))
    return {
        "total_provider_attempts": sum(
            item["provider_attempt_count"] for item in covered
        ),
        "cases_requiring_retry_count": sum(
            item["retry_count"] > 0 for item in covered
        ),
        "total_retry_count": sum(item["retry_count"] for item in covered),
        "total_rate_limit_response_count": sum(
            item["rate_limit_response_count"] for item in covered
        ),
        "total_rate_limit_retry_count": sum(
            item["rate_limit_retry_count"] for item in covered
        ),
        "total_other_transient_retry_count": sum(
            item["other_transient_retry_count"] for item in covered
        ),
        "total_schema_output_retry_count": sum(
            item["schema_output_retry_count"] for item in covered
        ),
        "total_pacing_sleep_seconds": round(
            sum(item["pacing_sleep_seconds"] for item in covered), 6
        ),
        "total_retry_backoff_sleep_seconds": round(
            sum(item["retry_backoff_sleep_seconds"] for item in covered), 6
        ),
        "telemetry_covered_selected_case_count": len(covered),
    }


def reference_aggregates(selected_scores):
    distribution = Counter()
    matched_types = Counter(
        {kind: 0 for kind in ("extractive", "abstractive", "boolean", "none")}
    )
    successful_references = 0
    reference_errors = 0
    non_first_winners = 0
    for record in selected_scores.values():
        summary = validate_reference_evaluations(record)
        distribution[str(summary["reference_count"])] += 1
        successful_references += summary["successful_reference_count"]
        reference_errors += summary["reference_error_count"]
        non_first_winners += summary["winner_non_first"]
        if summary["matched_answer_type"] is not None:
            matched_types[summary["matched_answer_type"]] += 1
    return {
        "successful_reference_judgment_count": successful_references,
        "reference_judgment_error_count": reference_errors,
        "reference_count_distribution": dict(sorted(distribution.items())),
        "cases_matched_by_answer_type": dict(matched_types),
        "non_first_winning_reference_count": non_first_winners,
    }


def update_manifest_attempt_counts(manifest, attempts):
    selected_scores = select_latest_scores(attempts)
    manifest["counts"]["score_attempt_records"] = sum(
        len(records) for records in attempts.values()
    )
    manifest["counts"]["completed_resume_keys"] = sum(
        record["status"] in {"success", "generation_error"}
        for record in selected_scores.values()
    )
    manifest["counts"]["judge_error_attempt_count"] = sum(
        record["status"] == "judge_error"
        for records in attempts.values()
        for record in records
    )
    manifest["counts"]["unresolved_judge_error_count"] = sum(
        record["status"] == "judge_error"
        for record in selected_scores.values()
    )
    manifest["counts"]["successful_judgments"] = sum(
        record["status"] == "success"
        for record in selected_scores.values()
    )
    manifest["counts"].update(reliability_aggregates(selected_scores))
    manifest["counts"].update(reference_aggregates(selected_scores))
    return selected_scores


def aggregate_metrics(
    run_id,
    judge_id,
    judge_model,
    threshold,
    reasoning_effort,
    max_retries,
    selected_cases,
    selected_scores,
    answers_path,
    started_at,
    completed_at,
    profile,
    model_ids,
    attempts=None,
    case_selection=None,
):
    scores = [
        selected_scores.get(
            resume_key(
                record["paper_id"],
                record["question_id"],
                judge_model,
                EVALUATOR_VERSION,
            )
        )
        for record in selected_cases
    ]
    available_scores = [record for record in scores if record is not None]
    non_null = [
        record for record in available_scores
        if record.get("semantic_score") is not None
    ]
    semantic_scores = [record["semantic_score"] for record in non_null]
    selected_count = len(selected_cases)
    non_null_count = len(non_null)
    coverage = non_null_count / selected_count if selected_count else 0.0
    mean_score = (
        sum(semantic_scores) / non_null_count if non_null_count else None
    )
    pass_rate = (
        sum(bool(record["passed"]) for record in non_null) / non_null_count
        if non_null_count
        else None
    )

    bands = Counter({"0-1": 0, "2-4": 0, "5-6": 0, "7-8": 0, "9-10": 0})
    for score in semantic_scores:
        if score <= 0.1:
            bands["0-1"] += 1
        elif score <= 0.4:
            bands["2-4"] += 1
        elif score <= 0.6:
            bands["5-6"] += 1
        elif score <= 0.8:
            bands["7-8"] += 1
        else:
            bands["9-10"] += 1

    reliability = reliability_aggregates(selected_scores)
    historical_judge_errors = sum(
        record.get("status") == "judge_error"
        for records in (attempts or {}).values()
        for record in records
    )
    unresolved_judge_errors = sum(
        record.get("status") == "judge_error"
        for record in available_scores
    )
    reference_summary = reference_aggregates(selected_scores)
    expected_reference_judgments = sum(
        len(validate_gold_bundle(
            record["gold_answers"],
            f"{record['paper_id']}/{record['question_id']}",
        ))
        for record in selected_cases
        if record["status"] == "success"
    )

    return {
        "generation_run_id": run_id,
        "path_base": PATH_BASE,
        "evaluation_profile": profile_provenance(profile),
        "models": model_ids,
        "judge_id": judge_id,
        "judge_model": judge_model,
        "deepeval_version": package_version("deepeval"),
        "groq_sdk_version": package_version("groq"),
        "evaluator_version": EVALUATOR_VERSION,
        "deepeval_metric_name": METRIC_NAME,
        "evaluation_steps": EVALUATION_STEPS,
        "rubric": RUBRIC_DEFINITION,
        "qasper_placeholder_policy": QASPER_PLACEHOLDER_POLICY,
        "qasper_annotation_evidence_policy": ANNOTATION_EVIDENCE_POLICY,
        "case_selection": case_selection,
        "threshold": threshold,
        "reasoning_effort": reasoning_effort,
        "max_retries": max_retries,
        "transport_reliability": {
            **reliability,
            "unresolved_judge_error_count": unresolved_judge_errors,
            "historical_judge_error_attempt_count": historical_judge_errors,
        },
        "selected_case_count": selected_count,
        "successful_generation_count": sum(
            record["status"] == "success" for record in selected_cases
        ),
        "generation_error_count": sum(
            record["status"] != "success" for record in selected_cases
        ),
        "successful_judgment_count": sum(
            record.get("status") == "success" for record in available_scores
        ),
        "judge_error_count": sum(
            record.get("status") == "judge_error" for record in available_scores
        ),
        "expected_reference_judgment_count": expected_reference_judgments,
        **reference_summary,
        "non_null_evaluated_case_count": non_null_count,
        "evaluation_coverage": coverage,
        "mean_semantic_correctness": mean_score,
        "pass_rate": pass_rate,
        "full_run_semantic_correctness": (
            mean_score
            if coverage == 1.0 and case_selection is None
            else None
        ),
        "score_counts_by_rubric_band": dict(bands),
        "start_timestamp": started_at,
        "completion_timestamp": completed_at,
        "source_answers_path": serialize_artifact_path(answers_path),
        "nondeterminism_note": (
            "This is a nondeterministic LLM-as-a-judge evaluation even with "
            "fixed prompts and deterministic provider settings."
        ),
        "reference_semantics_note": (
            "Each prediction is evaluated independently against every valid "
            "QASPER annotation, and the maximum reference-level semantic "
            "score is retained. Each request uses that annotation's gold "
            "evidence. QASPER cross-reference placeholders are treated as "
            "non-semantic annotation markup, and absence from an abbreviated "
            "reference is not itself evidence of fabrication."
        ),
        "log_probability_weighting_used": False,
        "log_probability_note": (
            "Disabled: the DeepEval 4.2 custom-model fallback uses structured "
            "JSON output from Groq and does not expose the raw-response "
            "log-probability interface required for weighted G-Eval."
        ),
        "evaluator_revision_note": (
            "v1.6 corrects judge instructions and adds annotation-specific "
            "gold evidence while preserving independent-reference maximum "
            "aggregation and all scoring thresholds."
        ),
    }


def make_judge_manifest(
    run_id,
    judge_id,
    judge_model,
    threshold,
    max_questions,
    model_configuration,
    max_retries,
    run_dir,
    judge_dir,
    selected_cases,
    duplicate_generation_attempt_count,
    profile,
    model_ids,
    case_selection=None,
):
    request_parameters = model_configuration["request_parameters"]
    return {
        "generation_run_id": run_id,
        "judge_id": judge_id,
        "judge_model": judge_model,
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "evaluator_version": EVALUATOR_VERSION,
        "deepeval_metric_name": METRIC_NAME,
        "deepeval_version": package_version("deepeval"),
        "groq_sdk_version": package_version("groq"),
        "path_base": PATH_BASE,
        "evaluation_profile": profile_provenance(profile),
        "models": model_ids,
        "judge_request": {
            "threshold": threshold,
            "max_questions": max_questions,
            "reasoning_effort": model_configuration["reasoning_effort"],
            "max_retries": max_retries,
            "request_interval_seconds": profile.judge_request_interval_seconds,
            "retry_base_delay_seconds": profile.judge_retry_base_delay_seconds,
            "retry_max_delay_seconds": profile.judge_retry_max_delay_seconds,
            "retry_jitter_seconds": profile.judge_retry_jitter_seconds,
            "max_retry_after_seconds": profile.judge_max_retry_after_seconds,
            "temperature": request_parameters["temperature"],
            "seed": request_parameters["seed"],
            "max_completion_tokens": request_parameters[
                "max_completion_tokens"
            ],
            "reasoning_format": request_parameters["reasoning_format"],
            "reasoning_output_control": "reasoning_format",
            "stream": request_parameters["stream"],
            "timeout": request_parameters["timeout"],
            "strict_mode": False,
            "provider_schema_mode": (
                "strict_json_schema"
                if judge_model == "qwen/qwen3.8-27b"
                else "json_object"
            ),
            "provider_schema_strict": judge_model == "qwen/qwen3.8-27b",
            "async_mode": False,
            "log_probability_weighting_used": False,
        },
        "evaluation_steps": EVALUATION_STEPS,
        "rubric": RUBRIC_DEFINITION,
        "qasper_placeholder_policy": QASPER_PLACEHOLDER_POLICY,
        "qasper_annotation_evidence_policy": ANNOTATION_EVIDENCE_POLICY,
        "case_selection": case_selection,
        "reference_semantics_note": (
            "Each prediction is evaluated independently against every valid "
            "QASPER annotation, and the maximum reference-level semantic "
            "score is retained. Each request uses that annotation's gold "
            "evidence. QASPER cross-reference placeholders are treated as "
            "non-semantic annotation markup, and absence from an abbreviated "
            "reference is not itself evidence of fabrication."
        ),
        "paths": {
            "generation_run": serialize_artifact_path(run_dir),
            "judge_run": serialize_artifact_path(judge_dir),
            "scores": serialize_artifact_path(judge_dir / "scores.jsonl"),
            "metrics": serialize_artifact_path(judge_dir / "metrics.json"),
        },
        "counts": {
            "selected_cases": len(selected_cases),
            "expected_reference_judgment_count": sum(
                len(validate_gold_bundle(
                    record["gold_answers"],
                    f"{record['paper_id']}/{record['question_id']}",
                ))
                for record in selected_cases
                if record["status"] == "success"
            ),
            "duplicate_generation_attempts": duplicate_generation_attempt_count,
            "score_attempt_records": 0,
            "completed_resume_keys": 0,
            "judge_error_attempt_count": 0,
            "unresolved_judge_error_count": 0,
            "successful_judgments": 0,
            **reliability_aggregates({}),
            **reference_aggregates({}),
        },
        "evaluator_revision_note": (
            "v1.6 corrects judge instructions and adds annotation-specific "
            "gold evidence; independent-reference maximum scoring is retained."
        ),
    }


def validate_resume_manifest(
    manifest,
    run_id,
    judge_id,
    judge_model,
    threshold,
    max_questions,
    reasoning_effort,
    max_retries,
    profile,
    scores_path,
    case_selection=None,
):
    expected = {
        "generation_run_id": run_id,
        "judge_id": judge_id,
        "judge_model": judge_model,
        "evaluator_version": EVALUATOR_VERSION,
    }
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"Resume judge identity mismatch: expected {expected}, found {actual}."
        )
    if manifest.get("case_selection") != case_selection:
        raise ValueError(
            "Resume case-selection provenance does not match generation."
        )
    if (
        manifest.get("evaluation_steps") != EVALUATION_STEPS
        or manifest.get("rubric") != RUBRIC_DEFINITION
        or manifest.get("qasper_placeholder_policy")
        != QASPER_PLACEHOLDER_POLICY
        or manifest.get("qasper_annotation_evidence_policy")
        != ANNOTATION_EVIDENCE_POLICY
    ):
        raise ValueError(
            "Resume judge-input methodology provenance is incompatible."
        )
    expected_configuration = {
        "threshold": threshold,
        "max_questions": max_questions,
        "reasoning_effort": reasoning_effort,
        "max_retries": max_retries,
        "request_interval_seconds": profile.judge_request_interval_seconds,
        "retry_base_delay_seconds": profile.judge_retry_base_delay_seconds,
        "retry_max_delay_seconds": profile.judge_retry_max_delay_seconds,
        "retry_jitter_seconds": profile.judge_retry_jitter_seconds,
        "max_retry_after_seconds": profile.judge_max_retry_after_seconds,
    }
    configuration = manifest.get("judge_request")
    if configuration is None:
        configuration = manifest.get("configuration", {})
    actual_configuration = {
        key: configuration.get(key) for key in expected_configuration
    }
    if actual_configuration != expected_configuration:
        raise ValueError(
            "Resume configuration must match the frozen judge manifest: "
            f"expected {actual_configuration}, received {expected_configuration}."
        )
    return validate_profile_provenance(
        manifest,
        profile,
        records_path=scores_path,
        allow_legacy_read=False,
    )


def resolve_effective_judge_profile(
    generation_profile,
    judge_dir,
    resume,
    requested_model=None,
    requested_threshold=None,
    requested_reasoning_effort=None,
    requested_max_retries=None,
):
    manifest_path = Path(judge_dir) / "judge_manifest.json"
    snapshot_path = Path(judge_dir) / LEGACY_RESOLVED_CONFIG_FILENAME
    if resume and (manifest_path.is_file() or snapshot_path.is_file()):
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                recorded_manifest = json.load(handle)
            if recorded_manifest.get("evaluator_version") != EVALUATOR_VERSION:
                raise ValueError(
                    "Judge resume uses an incompatible evaluator/artifact "
                    "version. Start v1.6 with a new judge ID."
                )
            if recorded_manifest.get("evaluation_profile") is not None:
                effective_profile = config_from_provenance(
                    recorded_manifest
                )
            elif snapshot_path.is_file():
                effective_profile = load_legacy_evaluation_config(
                    snapshot_path
                )
            else:
                raise ValueError(
                    "Judge resume lacks embedded configuration provenance."
                )
        else:
            effective_profile = load_legacy_evaluation_config(snapshot_path)
        validate_generation_profile_scope(
            effective_profile,
            generation_profile,
        )
        explicit_fields = {
            "judge_model": requested_model,
            "judge_threshold": requested_threshold,
            "judge_reasoning_effort": requested_reasoning_effort,
            "judge_max_retries": requested_max_retries,
        }
        mismatches = {
            field: (getattr(effective_profile, field), value)
            for field, value in explicit_fields.items()
            if value is not None
            and getattr(effective_profile, field) != value
        }
        if mismatches:
            raise ValueError(
                "Judge resume overrides do not match the judge-specific "
                f"resolved configuration: {mismatches}."
            )
        return effective_profile

    judge_model = requested_model or generation_profile.judge_model
    reasoning_effort = requested_reasoning_effort
    if (
        reasoning_effort is None
        and requested_model is not None
        and requested_model != generation_profile.judge_model
    ):
        reasoning_effort = default_reasoning_effort(
            judge_model,
            fallback=generation_profile.judge_reasoning_effort,
        )
    return generation_profile.with_judge_overrides(
        model=judge_model,
        threshold=requested_threshold,
        reasoning_effort=reasoning_effort,
        max_retries=requested_max_retries,
    )


def run_evaluation(args):
    run_id = args.run_id
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("Invalid generation run ID.")

    run_dir = RUNS_DIR / run_id
    generation_profile = load_evaluation_config()
    requested_model = getattr(args, "judge_model", None)
    provisional_model = requested_model or generation_profile.judge_model
    judge_id = args.judge_id or safe_judge_id(provisional_model)
    if not JUDGE_ID_RE.fullmatch(judge_id):
        raise ValueError("Invalid judge ID.")
    judge_dir = run_dir / "deepeval" / judge_id
    effective_profile = resolve_effective_judge_profile(
        generation_profile,
        judge_dir,
        args.resume,
        requested_model=requested_model,
        requested_threshold=getattr(args, "threshold", None),
        requested_reasoning_effort=getattr(args, "reasoning_effort", None),
        requested_max_retries=getattr(args, "max_retries", None),
    )
    args.judge_model = effective_profile.judge_model
    args.threshold = effective_profile.judge_threshold
    args.reasoning_effort = effective_profile.judge_reasoning_effort
    args.max_retries = effective_profile.judge_max_retries
    model_configuration = resolve_model_configuration(
        args.judge_model,
        args.reasoning_effort,
        effective_profile.judge_max_completion_tokens,
    )
    run_dir, answers_path, selected_cases, duplicate_count = prepare_cases(
        run_id,
        args.max_questions,
        profile=generation_profile,
    )
    scores_path = judge_dir / "scores.jsonl"
    metrics_path = judge_dir / "metrics.json"
    manifest_path = judge_dir / "judge_manifest.json"
    generation_manifest, _ = load_generation_manifest(
        run_dir / "generation_manifest.json",
        run_id,
    )
    case_selection = generation_manifest.get("case_selection")
    model_ids = dict(generation_manifest.get("models") or {})
    model_ids["judge"] = args.judge_model

    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume; judge manifest not found: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        has_profile_provenance = validate_resume_manifest(
            manifest,
            run_id,
            judge_id,
            args.judge_model,
            args.threshold,
            args.max_questions,
            model_configuration["reasoning_effort"],
            args.max_retries,
            effective_profile,
            scores_path,
            case_selection,
        )
        if not has_profile_provenance:
            manifest["path_base"] = PATH_BASE
            manifest["evaluation_profile"] = profile_provenance(
                effective_profile
            )
            manifest["models"] = model_ids
        manifest["status"] = "running"
        manifest["completed_at"] = None
    else:
        if judge_dir.exists():
            raise FileExistsError(
                f"Judge run already exists: {judge_dir}. Use --resume to continue it."
            )
        judge_dir.mkdir(parents=True)
        manifest = make_judge_manifest(
            run_id,
            judge_id,
            args.judge_model,
            args.threshold,
            args.max_questions,
            model_configuration,
            args.max_retries,
            run_dir,
            judge_dir,
            selected_cases,
            duplicate_count,
            effective_profile,
            model_ids,
            case_selection,
        )
    write_json(manifest_path, manifest)

    attempts = load_score_attempts(scores_path, args.judge_model)
    validate_attempt_annotation_evidence(
        attempts,
        selected_cases,
        args.judge_model,
    )
    selected_scores = select_latest_scores(attempts)
    terminal_keys = {
        key
        for key, record in selected_scores.items()
        if record["status"] in {"success", "generation_error"}
    }
    pending_successes = [
        record
        for record in selected_cases
        if record["status"] == "success"
        and resume_key(
            record["paper_id"],
            record["question_id"],
            args.judge_model,
            EVALUATOR_VERSION,
        ) not in terminal_keys
    ]

    started_at = manifest["started_at"]
    try:
        metric = None
        judge_model = None
        if pending_successes:
            load_dotenv(REPO_ROOT / ".env")
            judge_model = GroqJudgeModel(
                model_configuration=model_configuration,
                max_retries=args.max_retries,
                request_interval_seconds=(
                    effective_profile.judge_request_interval_seconds
                ),
                retry_base_delay_seconds=(
                    effective_profile.judge_retry_base_delay_seconds
                ),
                retry_max_delay_seconds=(
                    effective_profile.judge_retry_max_delay_seconds
                ),
                retry_jitter_seconds=(
                    effective_profile.judge_retry_jitter_seconds
                ),
                max_retry_after_seconds=(
                    effective_profile.judge_max_retry_after_seconds
                ),
            )
            metric = build_metric(judge_model, args.threshold)

        for record in selected_cases:
            key = resume_key(
                record["paper_id"],
                record["question_id"],
                args.judge_model,
                EVALUATOR_VERSION,
            )
            if key in terminal_keys:
                continue
            previous_attempts = attempts.get(key, [])
            score = evaluate_case(
                record,
                metric,
                args.judge_model,
                judge_id,
                args.threshold,
                len(previous_attempts) + 1,
                telemetry_source=judge_model,
            )
            append_jsonl(scores_path, score)
            attempts.setdefault(key, []).append(score)
            selected_scores = update_manifest_attempt_counts(
                manifest,
                attempts,
            )
            if score["status"] in {"success", "generation_error"}:
                terminal_keys.add(key)
            write_json(manifest_path, manifest)

        completed_at = utc_now()
        metrics = aggregate_metrics(
            run_id,
            judge_id,
            args.judge_model,
            args.threshold,
            model_configuration["reasoning_effort"],
            args.max_retries,
            selected_cases,
            select_latest_scores(attempts),
            answers_path,
            started_at,
            completed_at,
            effective_profile,
            model_ids,
            attempts=attempts,
            case_selection=case_selection,
        )
        write_json(metrics_path, metrics)
        manifest["status"] = (
            "complete"
            if metrics["judge_error_count"] == 0
            else "complete_with_judge_errors"
        )
        manifest["completed_at"] = completed_at
        update_manifest_attempt_counts(manifest, attempts)
        manifest["counts"].update(
            {
                "generation_errors": metrics["generation_error_count"],
                "non_null_evaluated_cases": metrics[
                    "non_null_evaluated_case_count"
                ],
                "expected_reference_judgment_count": metrics[
                    "expected_reference_judgment_count"
                ],
                "successful_reference_judgment_count": metrics[
                    "successful_reference_judgment_count"
                ],
                "reference_judgment_error_count": metrics[
                    "reference_judgment_error_count"
                ],
                "reference_count_distribution": metrics[
                    "reference_count_distribution"
                ],
                "cases_matched_by_answer_type": metrics[
                    "cases_matched_by_answer_type"
                ],
                "non_first_winning_reference_count": metrics[
                    "non_first_winning_reference_count"
                ],
            }
        )
        write_json(manifest_path, manifest)
    except BaseException:
        manifest["status"] = "interrupted"
        manifest["completed_at"] = utc_now()
        write_json(manifest_path, manifest)
        raise

    print(
        f"Judge run {judge_id} finished with status {manifest['status']}; "
        f"coverage={metrics['evaluation_coverage']:.2%}."
    )
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate QASPER answer correctness with DeepEval G-Eval on Groq."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--judge-model",
        help="Override the judge model from the Python evaluation settings.",
    )
    parser.add_argument("--judge-id")
    parser.add_argument(
        "--threshold",
        type=threshold_value,
        help="Override the judge threshold from the Python evaluation settings.",
    )
    parser.add_argument("--max-questions", type=positive_int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "default", "low", "medium", "high"),
    )
    parser.add_argument(
        "--max-retries",
        type=nonnegative_int,
        help="Override retries from the Python evaluation settings.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_evaluation(args)


if __name__ == "__main__":
    main()
