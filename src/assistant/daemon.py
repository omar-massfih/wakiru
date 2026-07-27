"""The daemon — the assistant as a living agent, with no web surface.

The process is nothing but the background loops and the chat channels: the
heartbeat (the single entry point for reminders, digests, and data
refreshes), the nightly sleep pass, Telegram long-polling, and Slack socket
mode. There is no REST API and no web UI; everything that used to be an
endpoint is a Telegram command (``/heartbeat``, ``/briefing``, ``/review``,
``/sync``, ``/sleep``) or simply the heartbeat's job. The one HTTP remnant
is a bare unauthenticated ``GET /health`` liveness endpoint on
``HEALTH_PORT``, so container orchestrators can keep probing the process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import threading
from collections.abc import Callable
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import slack, telegram
from .agent import build_agent
from .calendar.context import now
from .config import get_settings
from .heartbeat import next_wake_at as heartbeat_next_wake
from .heartbeat import run_heartbeat
from .sleep import run_sleep

logger = logging.getLogger(__name__)


@lru_cache
def _agent():
    """Build the graph once and share it across every loop and channel."""
    return build_agent()


async def _ticker(label: str, worker: Callable[[], object], interval: Callable[[], float]) -> None:
    """Run ``worker`` forever on a wall-clock cadence, independent of chat traffic.

    The workers are synchronous (SQLite + urllib), so each tick runs in a worker
    thread to keep the event loop free. Best-effort: any error is logged and the
    loop keeps ticking; the workers' own dedupe ledgers / idempotent upserts make
    every pass safe to repeat. ``interval`` is a callable so a tick always sleeps
    on the current settings.
    """
    while True:
        try:
            await asyncio.to_thread(worker)
        except Exception:
            logger.exception("%s tick failed", label)
        await asyncio.sleep(interval())


async def _heartbeat_loop() -> None:
    """Wake the heartbeat when it is due — the one scheduler for all activity.

    Not a fixed-cadence sleep: each tick asks ``heartbeat.next_wake_at`` when the
    next wake should be (the fixed ``heartbeat_minutes`` by default, pulled
    earlier by a soon-due follow-up, an opening reminder band, a scheduled
    digest, a due data refresh, or a model-set ``set_next_wake``), and only
    then wakes the model. Sleeps in slices of at most 60s so a follow-up or
    self-wake scheduled mid-sleep (from a chat turn) takes effect within the
    minute. The slice is a cheap SQLite read, not a model call; every wake is
    the token-cost dial. Each wake first runs the due refreshes (mail,
    weather, calendar sync) and then — heartbeat enabled, outside quiet
    hours — the deliberative model call.
    """
    while True:
        try:
            settings = get_settings()
            current = now(settings)
            target = await asyncio.to_thread(heartbeat_next_wake, settings, current)
            if current >= target:
                await asyncio.to_thread(run_heartbeat, settings, _agent())
                delay = 60.0
            else:
                delay = min((target - current).total_seconds(), 60.0)
        except Exception:
            logger.exception("heartbeat tick failed")
            delay = 60.0
        await asyncio.sleep(max(delay, 1.0))


async def _sleep_loop() -> None:
    """Run the nightly memory-maintenance pass on its own slow cadence.

    Its own loop, not the heartbeat: it must run even with the heartbeat off,
    and its consolidation LLM step can take minutes — riding a wake would delay
    it. The once-per-day ledger makes every tick outside the due window a cheap
    no-op, so a 5-minute cadence just bounds how late after ``sleep_time`` the
    pass lands.
    """
    await _ticker("sleep", lambda: run_sleep(get_settings(), _agent()), lambda: 300)


def _log_task_death(task: asyncio.Task) -> None:
    """Surface a background task that stopped on its own — it should run forever."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("background task %r died", task.get_name(), exc_info=exc)
    else:
        logger.error("background task %r exited unexpectedly", task.get_name())


class _HealthHandler(BaseHTTPRequestHandler):
    """``GET /health`` → 200, anything else → 404. No auth: it leaks nothing."""

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/health":
            self.send_error(404)
            return
        body = json.dumps({"status": "ok"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        pass  # probes every few seconds would drown the real logs


def _start_health_server(port: int) -> ThreadingHTTPServer | None:
    """Serve the liveness endpoint from a daemon thread; ``None`` when disabled."""
    if port <= 0:
        return None
    server = ThreadingHTTPServer(("0.0.0.0", port), _HealthHandler)
    threading.Thread(target=server.serve_forever, name="health", daemon=True).start()
    logger.info("health endpoint on :%d/health", port)
    return server


async def _run() -> None:
    settings = get_settings()
    if settings.codex_sandbox != "read-only" and (
        settings.telegram_bot_token or settings.slack_bot_token
    ):
        logger.warning(
            "CODEX_SANDBOX=%s while a remote channel is configured (telegram/slack)"
            " — anyone who can message the assistant can make Codex write to the"
            " filesystem.",
            settings.codex_sandbox,
        )
    if settings.enable_code_execution and (
        settings.telegram_bot_token or settings.slack_bot_token
    ):
        logger.warning(
            "ENABLE_CODE_EXECUTION=1 while a remote channel is configured "
            "(telegram/slack) — anyone who can message the assistant can run "
            "Python in this container.",
        )

    health = _start_health_server(settings.health_port)
    tasks: list[asyncio.Task] = []
    # The heartbeat loop is the single entry point for all background activity
    # (reminders, digests, data refreshes) — started whenever a cadence is set,
    # even with the deliberative layer off, because the refreshes ride it too.
    if settings.heartbeat_minutes > 0:
        tasks.append(asyncio.create_task(_heartbeat_loop(), name="heartbeat"))
        logger.info(
            "heartbeat started (every %d min, deliberative layer %s)",
            settings.heartbeat_minutes,
            "on" if settings.enable_heartbeat else "off",
        )
    if settings.enable_sleep:
        tasks.append(asyncio.create_task(_sleep_loop(), name="sleep"))
        logger.info("nightly sleep started (due at %s)", settings.sleep_time)
    if settings.telegram_bot_token:
        tasks.append(
            asyncio.create_task(telegram.poll_loop(_agent(), settings), name="telegram-poll")
        )
    stop_socket_mode = None
    if settings.slack_app_token and settings.slack_bot_token:
        # A failed websocket connect must not take the whole daemon down —
        # the other channels still work without Slack.
        try:
            # to_thread: connect() blocks on the websocket handshake.
            stop_socket_mode = await asyncio.to_thread(
                slack.start_socket_mode, _agent(), settings
            )
            logger.info("slack socket mode connected")
        except Exception:
            logger.exception("slack socket mode failed to start; continuing without it")
    for task in tasks:
        task.add_done_callback(_log_task_death)
    if not tasks and stop_socket_mode is None:
        logger.warning("no loops or channels configured — the daemon is idle")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        await stop.wait()
    finally:
        if stop_socket_mode is not None:
            stop_socket_mode()
        for task in tasks:
            task.cancel()
        # Wait for cancellation to land so shutdown doesn't strand mid-operation
        # work; return_exceptions swallows the resulting CancelledErrors.
        await asyncio.gather(*tasks, return_exceptions=True)
        if health is not None:
            health.shutdown()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(_run())
