import json
import re
import time

from core.llm import get_llm
from core.models import groundedness_checker_request_settings


GROUNDEDNESS_SCHEMA = {
    "name": "answer_groundedness",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": [
                    "grounded",
                    "partially_grounded",
                    "not_grounded",
                ],
            },
            "explanation": {
                "type": "string",
            },
        },
        "required": [
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
    settings = groundedness_checker_request_settings(settings_override)

    configuration = {
        "model": settings["model"],
        "temperature": settings["temperature"],
        "seed": settings["seed"],
        "max_completion_tokens": settings["max_completion_tokens"],
        "reasoning_format": settings["reasoning_format"],
        "reasoning_effort": settings["reasoning_effort"],
        "response_format": {
            "type": "json_schema",
            "json_schema": GROUNDEDNESS_SCHEMA,
        },
    }

    return configuration


def check_groundedness(
    question,
    answer,
    context,
    settings_override=None,
):
    if not context.strip():
        return (
            "not_grounded",
            "No retrieved context was available.",
            None,
        )

    system_prompt = """
You evaluate answer groundedness for a RAG system.

Use only the supplied retrieved context. Do not use outside knowledge.

Labels:
- grounded: every important answer claim is supported by the context
- partially_grounded: the answer is partly supported but contains an
  unsupported, vague, or extrapolated claim
- not_grounded: the answer is mostly unsupported or contradicted

Keep the explanation to one short sentence.
""".strip()

    user_prompt = f"""
Question:
{question}

Retrieved context:
{context}

Answer:
{answer}
""".strip()

    start = time.perf_counter()

    try:
        settings = groundedness_checker_request_settings(settings_override)
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
                "The groundedness checker returned an invalid choice count.",
                latency,
            )

        choice = response.choices[0]

        if choice.finish_reason == "length":
            return (
                "unknown",
                "The groundedness checker exhausted its completion budget.",
                latency,
            )

        content = (choice.message.content or "").strip()

        if not content:
            return (
                "unknown",
                "The groundedness checker returned empty content.",
                latency,
            )

        result = json.loads(content)

        label = result.get("label")
        explanation = result.get("explanation")

        if label not in {
            "grounded",
            "partially_grounded",
            "not_grounded",
        }:
            return (
                "unknown",
                "The groundedness checker returned an invalid label.",
                latency,
            )

        if not isinstance(explanation, str) or not explanation.strip():
            return (
                "unknown",
                "The groundedness checker returned an empty explanation.",
                latency,
            )

        return (
            label,
            explanation.strip(),
            latency,
        )

    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return (
            "unknown",
            "Could not validate groundedness output: "
            + _sanitize_error(error),
            time.perf_counter() - start,
        )

    except Exception as error:
        return (
            "unknown",
            "Groundedness check failed: "
            + _sanitize_error(error),
            time.perf_counter() - start,
        )
