import json
import re
import time

from core.llm import get_llm
from core.models import relevance_checker_request_settings
from retrieval.formatting import build_structured_chunk_text


RELEVANCE_SCHEMA = {
    "name": "retrieval_relevance",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "integer",
                    "enum": [0, 1, 2],
                },
            },
            "label": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "explanation": {
                "type": "string",
            },
        },
        "required": [
            "scores",
            "label",
            "explanation",
        ],
        "additionalProperties": False,
    },
}


def _sanitize_error(error, limit=500):
    message = re.sub(r"\s+", " ", str(error)).strip()
    return message[:limit]


def _request_configuration(settings_override=None):
    settings = relevance_checker_request_settings(settings_override)

    configuration = {
        "model": settings["model"],
        "temperature": settings["temperature"],
        "seed": settings["seed"],
        "max_completion_tokens": settings["max_completion_tokens"],
        "reasoning_format": settings["reasoning_format"],
        "reasoning_effort": settings["reasoning_effort"],
        "response_format": {
            "type": "json_schema",
            "json_schema": RELEVANCE_SCHEMA,
        },
    }

    return configuration


def compute_retrieval_relevance(
    question,
    reranked_docs,
    settings_override=None,
):
    """
    Evaluate whether the final retrieved contexts are useful for answering
    the question.

    Returns:
        (
            relevance_label,
            relevance_score,
            retrieval_explanation,
            retrieval_eval_latency,
        )
    """

    if not reranked_docs:
        return (
            "low",
            0.0,
            "No retrieved context was available.",
            None,
        )

    contexts = []

    for index, document in enumerate(reranked_docs, start=1):
        structured_text = build_structured_chunk_text(document)
        contexts.append(
            f"[Context {index}]\n{structured_text}"
        )

    context_text = "\n\n".join(contexts)

    system_prompt = """
You evaluate retrieval quality for a RAG system.

Evaluate only whether the supplied contexts contain information useful for
answering the question. Do not evaluate a generated answer and do not use
outside knowledge.

Assign one score to every context:
- 2: directly relevant and useful
- 1: partially relevant or useful supporting information
- 0: irrelevant

Set the overall label:
- high: enough relevant information is present to answer the question
- medium: useful information exists, but it is incomplete or indirect
- low: insufficient relevant information is present

Keep the explanation to one short sentence.
""".strip()

    user_prompt = f"""
Question:
{question}

Retrieved contexts:
{context_text}
""".strip()

    start = time.perf_counter()

    try:
        settings = relevance_checker_request_settings(settings_override)
        request = _request_configuration(settings_override)
        request["messages"] = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        response = (
            get_llm()
            .with_options(
                max_retries=settings["max_retries"],
                timeout=settings["timeout_seconds"],
            )
            .chat.completions.create(**request)
        )

        latency = time.perf_counter() - start

        if len(response.choices) != 1:
            return (
                "unknown",
                None,
                "The relevance checker returned an invalid choice count.",
                latency,
            )

        choice = response.choices[0]

        if choice.finish_reason == "length":
            return (
                "unknown",
                None,
                "The relevance checker exhausted its completion budget.",
                latency,
            )

        content = (choice.message.content or "").strip()

        if not content:
            return (
                "unknown",
                None,
                "The relevance checker returned empty content.",
                latency,
            )

        result = json.loads(content)

        scores = result.get("scores")
        label = result.get("label")
        explanation = result.get("explanation")

        if (
            not isinstance(scores, list)
            or len(scores) != len(reranked_docs)
            or any(
                not isinstance(score, int)
                or isinstance(score, bool)
                or score not in {0, 1, 2}
                for score in scores
            )
        ):
            return (
                "unknown",
                None,
                "The relevance checker returned invalid context scores.",
                latency,
            )

        if label not in {"high", "medium", "low"}:
            return (
                "unknown",
                None,
                "The relevance checker returned an invalid label.",
                latency,
            )

        if not isinstance(explanation, str) or not explanation.strip():
            return (
                "unknown",
                None,
                "The relevance checker returned an empty explanation.",
                latency,
            )

        weights = [
            1.0,
            0.8,
            0.6,
            0.4,
            0.2,
        ][:len(scores)]

        weighted_score = sum(
            (score / 2.0) * weight
            for score, weight in zip(scores, weights)
        )

        relevance_score = weighted_score / sum(weights)

        return (
            label,
            relevance_score,
            explanation.strip(),
            latency,
        )

    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return (
            "unknown",
            None,
            "Could not validate retrieval relevance output: "
            + _sanitize_error(error),
            time.perf_counter() - start,
        )

    except Exception as error:
        return (
            "unknown",
            None,
            "Retrieval relevance check failed: "
            + _sanitize_error(error),
            time.perf_counter() - start,
        )
