import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone


_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str | None) -> None:
    _request_id.set(request_id)


def get_request_id() -> str | None:
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Attach request_id to every record for consistent formatting.
        record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", None),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    """
    Configure application-wide logging.

    Env vars:
    - LOG_LEVEL: DEBUG|INFO|WARNING|ERROR (default: INFO)
    - LOG_FORMAT: text|json (default: text)
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.getenv("LOG_FORMAT", "text").lower()

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers when running under reloaders / tests.
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if fmt == "json":
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | request_id=%(request_id)s | %(message)s"
                )
            )
        handler.addFilter(RequestIdFilter())
        root.addHandler(handler)

    # Reduce noisy frameworks by default; can still be overridden via LOG_LEVEL.
    logging.getLogger("werkzeug").setLevel(max(level, logging.WARNING))

