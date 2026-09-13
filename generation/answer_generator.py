import time

from core.llm import get_llm
from core.models import generation_request_settings
from generation.prompt import PROMPT
from retrieval.formatting import build_structured_chunk_text


class AnswerGenerationError(RuntimeError):
    """The provider did not return a complete, usable answer."""


def extract_provider_answer(response):
    choices = getattr(response, "choices", None)
    try:
        choice_count = len(choices)
    except TypeError as exc:
        raise AnswerGenerationError(
            "The provider response did not contain a usable choices list."
        ) from exc

    if choice_count != 1:
        raise AnswerGenerationError(
            "The provider response must contain exactly one answer choice."
        )

    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if (
        isinstance(finish_reason, str)
        and finish_reason.strip().lower() == "length"
    ):
        raise AnswerGenerationError(
            "The provider truncated the answer at the completion-token limit."
        )

    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise AnswerGenerationError(
            "The provider returned no substantive answer content."
        )
    return content


def generate_answer(
    question,
    chat_history,
    reranked_docs,
):
    context = "\n\n".join(
        build_structured_chunk_text(d)
        for d in reranked_docs
    )

    history = "".join(
        f"{m['role'].title()}: {m['content']}\n"
        for m in chat_history[-6:]
    )

    prompt = PROMPT.format(
        history=history,
        context=context,
        question=question,
    )

    request = generation_request_settings()

    start = time.time()

    response = get_llm().chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        **request,
    )

    llm_latency = time.time() - start

    answer = extract_provider_answer(response)

    tokens = getattr(
        getattr(response, "usage", None),
        "total_tokens",
        None,
    )

    return (
        answer,
        context,
        prompt,
        llm_latency,
        tokens,
    )
