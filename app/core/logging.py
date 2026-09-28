from __future__ import annotations

import logging
import sys


class _DatadogTraceContextFilter(logging.Filter):
    """Ensures dd.trace_id/dd.span_id always exist on log records.

    ddtrace-run injects these attributes when DD_LOGS_INJECTION=true.
    Outside that context (tests, scripts/start.sh, no active span),
    the attributes are simply absent -- this filter fills them in so
    the format string below never raises.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "dd.trace_id"):
            setattr(record, "dd.trace_id", "0")
        if not hasattr(record, "dd.span_id"):
            setattr(record, "dd.span_id", "0")
        return True


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    fmt = (
        '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", '
        '"message": "%(message)s", "dd.trace_id": "%(dd.trace_id)s", "dd.span_id": "%(dd.span_id)s"}'
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_DatadogTraceContextFilter())
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[handler],
    )
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
