"""Durable, evaluation-only artifact writers."""

import errno
import json
import os
import time
from pathlib import Path


UNSUPPORTED_FSYNC_ERRNOS = {
    value
    for value in (
        getattr(errno, "ENOSYS", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if value is not None
}


def _temporary_replace_error(error):
    if not isinstance(error, OSError):
        return False
    if getattr(error, "winerror", None) in {32, 33}:
        return True
    return error.errno == errno.EBUSY


def atomic_write_text(
    path,
    content,
    *,
    replace_attempts=5,
    retry_base_delay_seconds=0.05,
    sleep_fn=time.sleep,
    replace_fn=os.replace,
    fsync_fn=os.fsync,
):
    """Write beside the target, fsync, then replace with bounded lock retries.
    The ``.tmp`` file is intentionally retained if replacement is exhausted so a validated recovery can be attempted later.
    """
    path = Path(path)
    if not isinstance(content, str):
        raise TypeError("Atomic text content must be a string.")
    if (
        not isinstance(replace_attempts, int)
        or isinstance(replace_attempts, bool)
        or replace_attempts < 1
    ):
        raise ValueError("replace_attempts must be a positive integer.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        try:
            fsync_fn(handle.fileno())
        except OSError as error:
            if error.errno not in UNSUPPORTED_FSYNC_ERRNOS:
                raise
            # Explicitly unsupported virtual/network filesystems may fall
            # back to the successful buffered flush above.

    for attempt in range(replace_attempts):
        try:
            replace_fn(temporary, path)
            return
        except OSError as error:
            if not _temporary_replace_error(error) or attempt + 1 >= replace_attempts:
                raise
            sleep_fn(retry_base_delay_seconds * (2**attempt))


def atomic_write_json(path, payload, **kwargs):
    content = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"
    atomic_write_text(path, content, **kwargs)
