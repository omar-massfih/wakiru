"""Email subsystem tests — the disabled-by-default posture, read, draft, and the
independently-gated send.

No network: the IMAP/SMTP connection seams (``imap_connect`` / ``_smtp_connect``)
are monkeypatched with fakes, so header parsing, body extraction, draft building,
and every gate run for real while staying offline.
"""

from __future__ import annotations

from email.message import EmailMessage

import pytest

from assistant.config import Settings
from assistant.mail import client, context
from assistant.mail.client import MailDisabledError
from assistant.mail.oauth import MailAuthError


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        enable_email=True,
        email_address="me@example.com",
        email_auth="password",
        email_password="app-password",
    )


def _raw(sender: str, subject: str, date: str, body: str = "hello there") -> bytes:
    message = EmailMessage()
    message["From"] = sender
    message["Subject"] = subject
    message["Date"] = date
    message.set_content(body)
    return message.as_bytes()


class _FakeIMAP:
    """A minimal IMAP stand-in recording the commands the client issues."""

    def __init__(self, uids=(b"1", b"2"), raw=None):
        self.uids = list(uids)
        self.raw = raw or {}
        self.commands: list[tuple] = []
        self.appended: list[tuple] = []
        self.logged_out = False

    def select(self, mailbox, readonly=False):
        self.commands.append(("select", mailbox, readonly))

    def uid(self, command, *args):
        self.commands.append(("uid", command, *args))
        if command == "SEARCH":
            return "OK", [b" ".join(self.uids)]
        if command == "FETCH":
            uid = args[0]
            key = uid.decode() if isinstance(uid, bytes) else uid
            return "OK", [(b"header", self.raw[key])]
        raise AssertionError(command)

    def append(self, folder, flags, stamp, body):
        self.appended.append((folder, flags, body))

    def logout(self):
        self.logged_out = True


# --- the off switch --------------------------------------------------------- #


def test_disabled_by_default(tmp_path) -> None:
    s = Settings(memory_dir=str(tmp_path / "m"))
    assert s.enable_email is False
    with pytest.raises(MailDisabledError):
        client._require_enabled(s)


def test_enabled_without_address_raises(tmp_path) -> None:
    s = Settings(memory_dir=str(tmp_path / "m"), enable_email=True)
    with pytest.raises(MailAuthError):
        client._require_enabled(s)


def test_unread_summary_reports_disabled(tmp_path) -> None:
    s = Settings(memory_dir=str(tmp_path / "m"))
    assert "Email is off" in context.unread_summary(s)


# --- read paths ------------------------------------------------------------- #


def test_list_recent_parses_headers_newest_first(settings, monkeypatch) -> None:
    fake = _FakeIMAP(
        uids=(b"1", b"2"),
        raw={
            "1": _raw("Alice <a@x.com>", "First", "Mon, 1 Jan 2026 09:00:00 +0000"),
            "2": _raw("Bob <b@x.com>", "Second", "Mon, 2 Jan 2026 09:00:00 +0000"),
        },
    )
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)

    messages = client.list_recent(settings)
    assert [m.subject for m in messages] == ["Second", "First"]  # newest first
    assert messages[0].sender == "Bob <b@x.com>"
    # Read-only select + PEEK: listing must never mark mail as read.
    assert ("select", "INBOX", True) in fake.commands
    assert any("BODY.PEEK" in str(c) for c in fake.commands)
    assert fake.logged_out


def test_list_recent_searches_unseen_by_default(settings, monkeypatch) -> None:
    fake = _FakeIMAP(uids=(b"1",), raw={"1": _raw("a@x.com", "S", "D")})
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    client.list_recent(settings)
    assert ("uid", "SEARCH", None, "UNSEEN") in fake.commands
    client.list_recent(settings, unread_only=False)
    assert ("uid", "SEARCH", None, "ALL") in fake.commands


def test_list_recent_respects_limit(settings, monkeypatch) -> None:
    fake = _FakeIMAP(
        uids=(b"1", b"2", b"3"),
        raw={str(i): _raw("a@x.com", f"S{i}", "D") for i in (1, 2, 3)},
    )
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    assert len(client.list_recent(settings, limit=2)) == 2


def test_read_message_extracts_plain_body(settings, monkeypatch) -> None:
    fake = _FakeIMAP(raw={"7": _raw("a@x.com", "Subj", "D", body="the body text")})
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    message = client.read_message(settings, "7")
    assert message.subject == "Subj"
    assert "the body text" in message.body


def test_read_message_decodes_encoded_subject(settings, monkeypatch) -> None:
    fake = _FakeIMAP(raw={"7": _raw("a@x.com", "Møte på torsdag", "D")})
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    assert client.read_message(settings, "7").subject == "Møte på torsdag"


def test_unread_summary_lists_subjects(settings, monkeypatch) -> None:
    fake = _FakeIMAP(uids=(b"1",), raw={"1": _raw("Alice <a@x.com>", "Lunch?", "D")})
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    summary = context.unread_summary(settings)
    assert "1 unread" in summary and "Lunch?" in summary


def test_unread_summary_handles_mailbox_error(settings, monkeypatch) -> None:
    def boom(_s):
        raise OSError("connection refused")

    monkeypatch.setattr(client, "imap_connect", boom)
    assert "Couldn't reach your mailbox" in context.unread_summary(settings)


# --- draft (the default write) ---------------------------------------------- #


def test_save_draft_appends_to_drafts_folder(settings, monkeypatch) -> None:
    fake = _FakeIMAP()
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    summary = client.save_draft(settings, "bob@x.com", "Hi", "body")
    assert "drafted" in summary and "bob@x.com" in summary
    assert len(fake.appended) == 1
    folder, flags, raw = fake.appended[0]
    assert folder == settings.email_drafts_folder
    assert flags == r"\Draft"
    assert b"bob@x.com" in raw and b"Hi" in raw


def test_save_draft_rejects_bad_address(settings, monkeypatch) -> None:
    monkeypatch.setattr(client, "imap_connect", lambda s: _FakeIMAP())
    with pytest.raises(ValueError):
        client.save_draft(settings, "not-an-address", "Hi", "body")


# --- send (gated independently) ---------------------------------------------- #


def test_send_is_blocked_unless_explicitly_enabled(settings, monkeypatch) -> None:
    # enable_email is on, but enable_email_send is not: sending must not happen.
    monkeypatch.setattr(
        client, "_smtp_connect", lambda s: pytest.fail("must not open SMTP")
    )
    with pytest.raises(MailDisabledError, match="Sending is disabled"):
        client.send_message(settings, "bob@x.com", "Hi", "body")


def test_send_works_when_both_switches_are_on(settings, monkeypatch) -> None:
    sent: list[EmailMessage] = []

    class _FakeSMTP:
        def send_message(self, message):
            sent.append(message)

        def quit(self):
            pass

    enabled = settings.model_copy(update={"enable_email_send": True})
    monkeypatch.setattr(client, "_smtp_connect", lambda s: _FakeSMTP())
    summary = client.send_message(enabled, "bob@x.com", "Hi", "body")
    assert "sent" in summary
    assert sent[0]["To"] == "bob@x.com"
    assert sent[0]["From"] == "me@example.com"


def test_send_blocked_when_email_disabled_entirely(tmp_path) -> None:
    s = Settings(memory_dir=str(tmp_path / "m"), enable_email_send=True)
    with pytest.raises(MailDisabledError):
        client.send_message(s, "bob@x.com", "Hi", "body")
