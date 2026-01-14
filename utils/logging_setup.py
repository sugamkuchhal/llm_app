import json
import logging
import os
import sys
from collections import deque
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


class InMemoryLogBufferHandler(logging.Handler):
    """
    Keeps a ring buffer of recent log records for debugging without Cloud Logging export.
    """

    def __init__(self, capacity: int = 2000):
        super().__init__()
        self.capacity = max(10, int(capacity))
        self._buf: deque[dict] = deque(maxlen=self.capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "request_id": getattr(record, "request_id", None),
                "msg": record.getMessage(),
            }
            if record.exc_info and self.formatter:
                entry["exc_info"] = self.formatter.formatException(record.exc_info)
            self._buf.append(entry)
        except Exception:
            # Never let logging failures crash the app.
            return

    def get(self, *, request_id: str | None = None, limit: int = 500) -> list[dict]:
        limit = max(1, min(int(limit), self.capacity))
        items = list(self._buf)
        if request_id:
            items = [x for x in items if x.get("request_id") == request_id]
        return items[-limit:]


_LOG_BUFFER: InMemoryLogBufferHandler | None = None


def get_log_buffer() -> InMemoryLogBufferHandler | None:
    return _LOG_BUFFER


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

        # Optional in-memory log buffer.
        if os.getenv("ENABLE_LOG_BUFFER", "0") == "1":
            global _LOG_BUFFER
            _LOG_BUFFER = InMemoryLogBufferHandler(
                capacity=int(os.getenv("LOG_BUFFER_SIZE", "2000"))
            )
            _LOG_BUFFER.setFormatter(handler.formatter)
            _LOG_BUFFER.addFilter(RequestIdFilter())
            root.addHandler(_LOG_BUFFER)

    # Reduce noisy frameworks by default; can still be overridden via LOG_LEVEL.
    logging.getLogger("werkzeug").setLevel(max(level, logging.WARNING))

