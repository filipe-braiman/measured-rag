"""Evaluation-owned checker diagnostics and copied-log sanitization."""

import copy
import re


CHECKER_CAUSES = (
    "rate_limit_tpm",
    "rate_limit_tpd",
    "malformed_output",
    "timeout",
    "connection",
    "other_failure",
)

_SECRET_REPLACEMENTS = (
    (re.compile(r"\bgsk_[A-Za-z0-9_-]+\b", re.I), "<redacted>"),
    (re.compile(r"\bBearer\s+\S+", re.I), "Bearer <redacted>"),
    (re.compile(r"\borg_[A-Za-z0-9_-]+\b", re.I), "<redacted-organization>"),
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "<redacted-email>"),
    (re.compile(r"(?i)https?://(?:console\.)?groq\.com/\S*"), "<redacted-provider-url>"),
    (re.compile(r"(?i)\b[A-Z]:[\\/][^\r\n,;]+"), "<redacted-path>"),
    (re.compile(r"(?<!\w)/(?:home|Users|tmp)/[^\r\n,;]+"), "<redacted-path>"),
)


def classify_checker_failure(explanation):
    text = explanation if isinstance(explanation, str) else ""
    lowered = text.lower()
    if "429" in lowered and ("tokens per day" in lowered or "tpd" in lowered):
        return "rate_limit_tpd"
    if "429" in lowered and ("tokens per minute" in lowered or "tpm" in lowered):
        return "rate_limit_tpm"
    if any(marker in lowered for marker in (
        "invalid context-score", "invalid context score", "malformed",
        "schema", "could not parse", "parse failure",
    )):
        return "malformed_output"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "connection" in lowered or "connect error" in lowered:
        return "connection"
    return "other_failure"


def normalized_checker_failure(explanation):
    cause = classify_checker_failure(explanation)
    if cause == "malformed_output" and "context-score" in str(explanation).lower():
        return "Checker returned invalid context-score count."
    status = "HTTP 429" if cause.startswith("rate_limit_") else None
    suffix = f" ({status})" if status else ""
    return f"Checker request failed: {cause}{suffix}."


def _sanitize_sensitive_text(value):
    sanitized = value
    for pattern, replacement in _SECRET_REPLACEMENTS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def sanitize_application_log(record):
    """Sanitize only diagnostic/error fields in a copied application record."""
    sanitized = copy.deepcopy(record)
    retrieval = sanitized.get("retrieval")
    if isinstance(retrieval, dict):
        label = str(retrieval.get("retrieval_relevance", "")).lower()
        explanation = retrieval.get("relevance_explanation")
        if isinstance(explanation, str):
            retrieval["relevance_explanation"] = (
                normalized_checker_failure(explanation)
                if label == "unknown"
                else _sanitize_sensitive_text(explanation)
            )
    generation = sanitized.get("generation")
    if isinstance(generation, dict):
        label = str(generation.get("groundedness_label", "")).lower()
        explanation = generation.get("groundedness_explanation")
        if isinstance(explanation, str):
            generation["groundedness_explanation"] = (
                normalized_checker_failure(explanation)
                if label == "unknown"
                else _sanitize_sensitive_text(explanation)
            )
    return sanitized


def empty_checker_diagnostics():
    return {
        role: {
            "valid": 0,
            "unknown": 0,
            "unknown_by_cause": {cause: 0 for cause in CHECKER_CAUSES},
        }
        for role in ("relevance", "groundedness")
    }


def aggregate_checker_diagnostics(records):
    counts = empty_checker_diagnostics()
    for record in records:
        if not isinstance(record, dict):
            continue
        retrieval = record.get("retrieval") or {}
        generation = record.get("generation") or {}
        for role, label, explanation in (
            (
                "relevance",
                retrieval.get("retrieval_relevance"),
                retrieval.get("relevance_explanation"),
            ),
            (
                "groundedness",
                generation.get("groundedness_label"),
                generation.get("groundedness_explanation"),
            ),
        ):
            if str(label).lower() == "unknown":
                counts[role]["unknown"] += 1
                cause = classify_checker_failure(explanation)
                counts[role]["unknown_by_cause"][cause] += 1
            elif isinstance(label, str) and label.strip():
                counts[role]["valid"] += 1
    return counts
