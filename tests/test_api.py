"""API auth tests — bearer-token gating; /health stays open.

Only the auth dependency is exercised (via the cheap read endpoints), so no
Codex, network, or agent build is involved.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from assistant.api import app
from assistant.config import Settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    def make(api_token: str | None):
        settings = Settings(memory_dir=str(tmp_path / "memory"), api_token=api_token)
        monkeypatch.setattr("assistant.api.get_settings", lambda: settings)
        return TestClient(app)

    return make


def test_health_stays_open_with_token_set(client) -> None:
    assert client("sekrit").get("/health").status_code == 200


def test_endpoints_reject_missing_or_wrong_token(client) -> None:
    c = client("sekrit")
    assert c.get("/memory").status_code == 401
    assert c.get("/memory", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/calendar", headers={"Authorization": "sekrit"}).status_code == 401  # not Bearer
    assert c.post("/reminders/run").status_code == 401


def test_endpoints_accept_valid_token(client) -> None:
    c = client("sekrit")
    headers = {"Authorization": "Bearer sekrit"}
    assert c.get("/memory", headers=headers).status_code == 200
    assert c.get("/calendar", headers=headers).status_code == 200


def test_unset_token_keeps_legacy_open_behavior(client) -> None:
    c = client(None)
    assert c.get("/memory").status_code == 200
    assert c.get("/calendar").status_code == 200
