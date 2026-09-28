"""Datadog AI Guard integration.

Evaluates chat/RAG prompts (and, for chat, the model's reply) through
Datadog AI Guard as an additional, independent detection layer on top
of the existing defense pipeline and NeMo Guardrails. Observe-only by
default -- see ``AppConfig.defense.ai_guard_block`` -- so it never
silently defeats the attack labs this app exists to teach.

Two independent switches must both be "on" for a real call to fire:
the ``DD_AI_GUARD_ENABLED`` env var (Datadog's own master switch) and
``settings.defense.ai_guard_enabled`` (this app's config toggle). If
either is off, or ``ddtrace`` isn't installed, or credentials are
missing, this module no-ops to an ALLOW verdict with zero added
latency or risk -- required so the app runs unmodified for anyone who
clones this repo without Datadog configured.

Tool-call evaluation (Datadog AI Guard's "Tool Protection") is not
wired in: ``app/services/tool_registry.py`` and
``app/services/agent_service.py`` are unimplemented scaffolds today
with no live callers.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)

_client = None
_client_init_attempted = False


@dataclass
class AIGuardVerdict:
    allowed: bool
    action: str
    reason: str | None = None
    tags: list[str] = field(default_factory=list)


def _ai_guard_env_enabled() -> bool:
    return os.environ.get("DD_AI_GUARD_ENABLED", "").strip().lower() in ("1", "true", "yes")


def _get_client():
    """Lazily construct and cache the AI Guard client.

    Returns None if ddtrace isn't installed, credentials are missing,
    or client construction otherwise fails -- callers must treat None
    as "AI Guard unavailable" and no-op.
    """
    global _client, _client_init_attempted
    if _client is not None:
        return _client
    if _client_init_attempted:
        return None
    _client_init_attempted = True
    try:
        from ddtrace.aiguard import new_ai_guard_client

        _client = new_ai_guard_client()
        logger.info("Datadog AI Guard client initialized")
    except Exception as exc:
        logger.warning("Datadog AI Guard client unavailable, evaluations will no-op: %s", exc)
        _client = None
    return _client


async def evaluate_prompt(
    messages: list[dict],
    *,
    block: bool,
    source: str,
) -> AIGuardVerdict:
    """Evaluate a conversation with Datadog AI Guard.

    ``messages`` follows the same ``[{"role": ..., "content": ...}]``
    shape already used across this app's Ollama client calls.
    ``source`` is a free-text label ("chat", "rag") for log filtering
    only, not passed to the SDK's own ``source``/telemetry parameter.
    """
    if not _ai_guard_env_enabled():
        return AIGuardVerdict(allowed=True, action="disabled")

    client = _get_client()
    if client is None:
        return AIGuardVerdict(allowed=True, action="disabled")

    try:
        from ddtrace.aiguard import AIGuardAbortError

        try:
            # evaluate() performs a blocking HTTP call; run it off the
            # event loop thread so a slow/unreachable AI Guard endpoint
            # can't stall request handling.
            result = await asyncio.to_thread(
                client.evaluate,
                messages,
                {"block": block},
            )
            verdict = AIGuardVerdict(
                allowed=result["action"] == "ALLOW",
                action=result["action"],
                reason=result.get("reason"),
                tags=result.get("tags") or [],
            )
        except AIGuardAbortError as exc:
            verdict = AIGuardVerdict(
                allowed=False,
                action=exc.action,
                reason=exc.reason,
                tags=exc.tags or [],
            )
    except Exception as exc:
        logger.warning("Datadog AI Guard evaluation failed (%s), allowing by default: %s", source, exc)
        return AIGuardVerdict(allowed=True, action="error")

    if verdict.allowed:
        logger.info("AI Guard [%s]: action=%s", source, verdict.action)
    else:
        logger.warning(
            "AI Guard [%s]: action=%s reason=%s tags=%s",
            source, verdict.action, verdict.reason, verdict.tags,
        )
    return verdict
