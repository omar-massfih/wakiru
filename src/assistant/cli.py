"""A terminal channel — chat with the assistant from a REPL.

The thinnest possible channel over the shared :mod:`assistant.chat` seam: read a
line from stdin, run one turn, print the reply, then run the same post-reply
upkeep every other channel does (memory, calendar, tasks, summarization). It uses
one stable thread id so a session's working memory and rolling summary persist
across turns (and across restarts, via the SQLite checkpointer). Run it with
``assistant-cli`` (or ``python -m assistant.cli``).
"""

from __future__ import annotations

import logging
import sys

from .agent import build_agent
from .chat import run_chat, run_upkeep
from .codex_runner import CodexError
from .config import get_settings

# A fixed thread so the conversation continues across turns and restarts. Distinct
# from the telegram:<chat_id> namespace so the two channels don't share history.
_THREAD_ID = "cli:default"


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    settings = get_settings()
    agent = build_agent(settings)
    print("Assistant ready. Type a message, or Ctrl-D / 'exit' to quit.")

    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message.lower() in ("exit", "quit"):
            break

        try:
            reply = run_chat(agent, message, _THREAD_ID, settings=settings)
        except CodexError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            continue
        print(f"bot> {reply}")

        # Same post-reply maintenance as the HTTP and Telegram channels. Inline
        # here (not backgrounded) since the REPL is single-user and sequential.
        try:
            run_upkeep(agent, settings, message, reply, _THREAD_ID)
        except Exception:
            logging.getLogger(__name__).exception("upkeep failed")


if __name__ == "__main__":
    main()
