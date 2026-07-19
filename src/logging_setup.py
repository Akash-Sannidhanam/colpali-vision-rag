"""Structured logging - one configured stdlib logger for the whole app.

Human-readable lines by default; set LOG_JSON=true for one JSON object per line
(handy for shipping to a log aggregator). Structured fields passed via the
standard `extra={...}` kwarg are rendered in both modes, e.g.

    log.info("gemini call", extra={"purpose": "answer", "total_tokens": 812})
"""

import json
import logging
import sys

from src import request_context
from src.config import LOG_JSON, LOG_LEVEL

# LogRecord attributes that are part of the record itself, not caller-supplied
# `extra=` fields. Everything else on the record is treated as a structured field.
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}

_configured = False


class _RequestIdFilter(logging.Filter):
    """Stamp the current `request_id` onto every record so a query's lines correlate.

    Attached to the single root handler, so gemini calls, node timings, and the
    degradation warnings all carry the same id. `request_id` isn't in `_RESERVED`, so
    `_Formatter` renders it automatically in both human and JSON modes.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_context.current_request_id()
        return True


class _Formatter(logging.Formatter):
    """Render records as human lines or JSON, including any `extra=` fields."""

    def __init__(self, json_mode: bool):
        super().__init__(datefmt="%H:%M:%S")
        self.json_mode = json_mode

    def format(self, record: logging.LogRecord) -> str:
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        if self.json_mode:
            payload = {
                "ts": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
                **extras,
            }
            if record.exc_info:
                payload["exc"] = self.formatException(record.exc_info)
            return json.dumps(payload, default=str)

        line = (f"{self.formatTime(record)} {record.levelname:<5} "
                f"{record.name} | {record.getMessage()}")
        if extras:
            line += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure() -> None:
    """Configure the root logger once, honoring LOG_LEVEL and LOG_JSON."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_Formatter(LOG_JSON))
    handler.addFilter(_RequestIdFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(LOG_LEVEL.upper())
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for `name` (configures logging on first use)."""
    configure()
    return logging.getLogger(name)
