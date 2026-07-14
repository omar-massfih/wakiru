"""Telegram pairing table for the Postgres backend."""

from __future__ import annotations

from ..config import Settings
from .core import (
    _schema_done,
    _schema_mark,
    connect,
)


def ensure_telegram_schema(settings: Settings) -> None:
    if _schema_done(settings, "telegram"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_telegram_chats (
              chat_id BIGINT PRIMARY KEY,
              paired_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
    _schema_mark(settings, "telegram")


def paired_telegram_chats(settings: Settings) -> list[int]:
    ensure_telegram_schema(settings)
    with connect(settings) as conn:
        rows = conn.execute("SELECT chat_id FROM assistant_telegram_chats ORDER BY chat_id").fetchall()
    return [int(row[0]) for row in rows]


def pair_telegram_chat(settings: Settings, chat_id: int) -> None:
    ensure_telegram_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_telegram_chats(chat_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (int(chat_id),),
        )
