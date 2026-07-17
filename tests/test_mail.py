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
    """A minimal IMAP stand-in recording the commands the client issues.

    ``capabilities`` defaults to a Gmail-shaped server; pass a string without
    ``X-GM-EXT-1`` to exercise the generic-IMAP paths. ``copy_fail`` makes the
    first N ``UID COPY`` attempts answer ``NO [TRYCREATE]``.
    """

    def __init__(
        self,
        uids=(b"1", b"2"),
        raw=None,
        capabilities=b"IMAP4REV1 UIDPLUS X-GM-EXT-1",
        copy_fail=0,
    ):
        self.uids = list(uids)
        self.raw = raw or {}
        self.capabilities = capabilities
        self.copy_fail = copy_fail
        self.commands: list[tuple] = []
        self.appended: list[tuple] = []
        self.created: list[str] = []
        self.expunged = False
        self.logged_out = False

    def select(self, mailbox, readonly=False):
        self.commands.append(("select", mailbox, readonly))

    def capability(self):
        return "OK", [self.capabilities]

    def uid(self, command, *args):
        self.commands.append(("uid", command, *args))
        if command == "SEARCH":
            return "OK", [b" ".join(self.uids)]
        if command == "FETCH":
            uid = args[0]
            key = uid.decode() if isinstance(uid, bytes) else uid
            if key not in self.raw:
                return "OK", [b""]
            return "OK", [(b"header", self.raw[key])]
        if command == "STORE":
            return "OK", [b""]
        if command == "COPY":
            if self.copy_fail > 0:
                self.copy_fail -= 1
                return "NO", [b"[TRYCREATE] no such folder"]
            return "OK", [b""]
        if command == "EXPUNGE":
            return "OK", [b""]
        raise AssertionError(command)

    def append(self, folder, flags, stamp, body):
        self.appended.append((folder, flags, body))

    def create(self, folder):
        self.created.append(folder)
        return "OK", [b""]

    def expunge(self):
        self.expunged = True
        return "OK", [b""]

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
    # It must not open IMAP either — the refusal saves nothing, so it must not
    # claim a draft was written (this message is surfaced verbatim to the user).
    monkeypatch.setattr(client, "imap_connect", lambda s: pytest.fail("must not draft"))
    with pytest.raises(MailDisabledError) as excinfo:
        client.send_message(settings, "bob@x.com", "Hi", "body")
    assert "nothing was sent or drafted" in str(excinfo.value)


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


# --- threaded replies -------------------------------------------------------- #


def _raw_original(
    sender="Alice <alice@x.com>",
    subject="Lunch?",
    message_id="<orig-123@x.com>",
    to=None,
    cc=None,
    reply_to=None,
    references=None,
) -> bytes:
    message = EmailMessage()
    message["From"] = sender
    message["Subject"] = subject
    if message_id:
        message["Message-ID"] = message_id
    if to:
        message["To"] = to
    if cc:
        message["Cc"] = cc
    if reply_to:
        message["Reply-To"] = reply_to
    if references:
        message["References"] = references
    message.set_content("original body")
    return message.as_bytes()


def _appended_draft(fake: _FakeIMAP) -> tuple[str, EmailMessage]:
    from email import message_from_bytes

    assert len(fake.appended) == 1
    folder, flags, raw = fake.appended[0]
    assert flags == r"\Draft"
    return folder, message_from_bytes(raw)


def test_reply_draft_threads_into_the_conversation(settings, monkeypatch) -> None:
    fake = _FakeIMAP(raw={"5": _raw_original(references="<older-1@x.com>")})
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)

    summary = client.save_reply_draft(settings, "5", "yes, 12:00 works")
    assert "reply drafted" in summary and "alice@x.com" in summary

    folder, draft = _appended_draft(fake)
    assert folder == settings.email_drafts_folder
    assert draft["To"] == "alice@x.com"
    assert draft["Subject"] == "Re: Lunch?"
    assert draft["In-Reply-To"] == "<orig-123@x.com>"
    assert draft["References"] == "<older-1@x.com> <orig-123@x.com>"
    assert draft["Message-ID"]  # a draft sent later must thread too
    # Headers were fetched with PEEK from a readonly select — drafting a reply
    # must not mark the original read.
    assert ("select", "INBOX", True) in fake.commands
    assert any("BODY.PEEK" in str(c) for c in fake.commands)


def test_reply_does_not_double_prefix_subject(settings, monkeypatch) -> None:
    for existing in ("Re: Lunch?", "SV: Lunsj?", "aw: Mittag?"):
        fake = _FakeIMAP(raw={"5": _raw_original(subject=existing)})
        monkeypatch.setattr(client, "imap_connect", lambda s, fake=fake: fake)
        client.save_reply_draft(settings, "5", "ok")
        _, draft = _appended_draft(fake)
        assert draft["Subject"] == existing


def test_reply_prefers_reply_to_over_from(settings, monkeypatch) -> None:
    fake = _FakeIMAP(raw={"5": _raw_original(reply_to="list-reply@x.com")})
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    client.save_reply_draft(settings, "5", "ok")
    _, draft = _appended_draft(fake)
    assert draft["To"] == "list-reply@x.com"


def test_reply_all_ccs_the_others_but_never_self(settings, monkeypatch) -> None:
    fake = _FakeIMAP(
        raw={
            "5": _raw_original(
                to="Me <me@example.com>, Bob <bob@x.com>", cc="carol@x.com"
            )
        }
    )
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    client.save_reply_draft(settings, "5", "ok", reply_all=True)
    _, draft = _appended_draft(fake)
    assert draft["Cc"] == "bob@x.com, carol@x.com"

    fake.appended.clear()
    client.save_reply_draft(settings, "5", "ok")  # plain reply: no Cc
    _, draft = _appended_draft(fake)
    assert draft["Cc"] is None


def test_reply_missing_uid_is_a_friendly_string(settings, monkeypatch) -> None:
    fake = _FakeIMAP(raw={})
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    assert client.save_reply_draft(settings, "9", "ok") == "No message with uid 9."
    assert fake.appended == []


def test_send_reply_blocked_unless_explicitly_enabled(settings, monkeypatch) -> None:
    monkeypatch.setattr(
        client, "_smtp_connect", lambda s: pytest.fail("must not open SMTP")
    )
    monkeypatch.setattr(
        client, "imap_connect", lambda s: pytest.fail("must not even fetch headers")
    )
    with pytest.raises(MailDisabledError) as excinfo:
        client.send_reply(settings, "5", "ok")
    assert "nothing was sent or drafted" in str(excinfo.value)


def test_send_reply_threads_when_both_switches_are_on(settings, monkeypatch) -> None:
    sent: list[EmailMessage] = []

    class _FakeSMTP:
        def send_message(self, message):
            sent.append(message)

        def quit(self):
            pass

    enabled = settings.model_copy(update={"enable_email_send": True})
    fake = _FakeIMAP(raw={"5": _raw_original()})
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    monkeypatch.setattr(client, "_smtp_connect", lambda s: _FakeSMTP())

    summary = client.send_reply(enabled, "5", "ok")
    assert "reply sent" in summary
    assert sent[0]["To"] == "alice@x.com"
    assert sent[0]["In-Reply-To"] == "<orig-123@x.com>"


# --- mailbox mutations: archive / mark read / label --------------------------- #


def test_archive_on_gmail_drops_the_inbox_label(settings, monkeypatch) -> None:
    fake = _FakeIMAP(raw={"5": _raw_original()})  # default caps advertise X-GM-EXT-1
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)

    summary = client.archive_message(settings, "5")
    assert "archived" in summary and "All Mail" in summary
    assert ("uid", "STORE", "5", "-X-GM-LABELS", r"(\Inbox)") in fake.commands
    # Gmail archive is label removal only — never a destructive copy/expunge.
    assert not any(c[1] == "COPY" for c in fake.commands if c[0] == "uid")
    assert ("select", "INBOX", False) in fake.commands  # a write needs write access


def test_archive_on_generic_imap_copies_before_deleting(settings, monkeypatch) -> None:
    fake = _FakeIMAP(raw={"5": _raw_original()}, capabilities=b"IMAP4REV1 UIDPLUS")
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)

    summary = client.archive_message(settings, "5")
    assert "moved to Archive" in summary
    uid_cmds = [c for c in fake.commands if c[0] == "uid" and c[1] != "FETCH"]
    kinds = [c[1] for c in uid_cmds]
    assert kinds.index("COPY") < kinds.index("STORE") < kinds.index("EXPUNGE")
    assert ("uid", "COPY", "5", '"Archive"') in fake.commands
    assert ("uid", "STORE", "5", "+FLAGS", r"(\Deleted)") in fake.commands


def test_archive_without_uidplus_uses_plain_expunge(settings, monkeypatch) -> None:
    fake = _FakeIMAP(raw={"5": _raw_original()}, capabilities=b"IMAP4REV1")
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    client.archive_message(settings, "5")
    assert fake.expunged
    assert not any(c[1] == "EXPUNGE" for c in fake.commands if c[0] == "uid")


def test_archive_creates_the_folder_once_and_retries(settings, monkeypatch) -> None:
    fake = _FakeIMAP(
        raw={"5": _raw_original()}, capabilities=b"IMAP4REV1 UIDPLUS", copy_fail=1
    )
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    summary = client.archive_message(settings, "5")
    assert "moved to Archive" in summary
    assert fake.created == ['"Archive"']


def test_archive_never_deletes_when_copy_keeps_failing(settings, monkeypatch) -> None:
    fake = _FakeIMAP(
        raw={"5": _raw_original()}, capabilities=b"IMAP4REV1 UIDPLUS", copy_fail=2
    )
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    with pytest.raises(RuntimeError):
        client.archive_message(settings, "5")
    assert ("uid", "STORE", "5", "+FLAGS", r"(\Deleted)") not in fake.commands


def test_archive_missing_uid_is_a_friendly_string(settings, monkeypatch) -> None:
    fake = _FakeIMAP(raw={})
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    assert client.archive_message(settings, "9") == "No message with uid 9."


def test_email_provider_override_beats_capabilities(settings, monkeypatch) -> None:
    # The server advertises X-GM-EXT-1 but the config says generic: move, not label.
    generic = settings.model_copy(update={"email_provider": "generic"})
    fake = _FakeIMAP(raw={"5": _raw_original()})
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    assert "moved to Archive" in client.archive_message(generic, "5")

    # And the reverse: no Gmail capability, but forced gmail.
    forced = settings.model_copy(update={"email_provider": "gmail"})
    fake2 = _FakeIMAP(raw={"5": _raw_original()}, capabilities=b"IMAP4REV1")
    monkeypatch.setattr(client, "imap_connect", lambda s: fake2)
    assert "All Mail" in client.archive_message(forced, "5")


def test_mark_read_sets_and_clears_seen(settings, monkeypatch) -> None:
    fake = _FakeIMAP(raw={"5": _raw_original()})
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)

    summary = client.mark_read(settings, "5")
    assert summary.startswith("marked read:") and "Lunch?" in summary
    assert ("uid", "STORE", "5", "+FLAGS", r"(\Seen)") in fake.commands

    summary = client.mark_read(settings, "5", unread=True)
    assert summary.startswith("marked unread:")
    assert ("uid", "STORE", "5", "-FLAGS", r"(\Seen)") in fake.commands


def test_label_on_gmail_quotes_and_removes(settings, monkeypatch) -> None:
    fake = _FakeIMAP(raw={"5": _raw_original()})
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)

    summary = client.set_label(settings, "5", "Receipts 2026")
    assert "labeled" in summary
    assert ("uid", "STORE", "5", "+X-GM-LABELS", '("Receipts 2026")') in fake.commands

    summary = client.set_label(settings, "5", "Receipts 2026", remove=True)
    assert "unlabeled" in summary
    assert ("uid", "STORE", "5", "-X-GM-LABELS", '("Receipts 2026")') in fake.commands


def test_label_on_generic_imap_moves_to_folder(settings, monkeypatch) -> None:
    fake = _FakeIMAP(raw={"5": _raw_original()}, capabilities=b"IMAP4REV1 UIDPLUS")
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    summary = client.set_label(settings, "5", "Receipts")
    assert "moved to folder" in summary
    assert ("uid", "COPY", "5", '"Receipts"') in fake.commands


def test_label_remove_on_generic_imap_explains_itself(settings, monkeypatch) -> None:
    fake = _FakeIMAP(raw={"5": _raw_original()}, capabilities=b"IMAP4REV1 UIDPLUS")
    monkeypatch.setattr(client, "imap_connect", lambda s: fake)
    summary = client.set_label(settings, "5", "Receipts", remove=True)
    assert summary.startswith("This server has folders")
    assert not any(c[1] == "COPY" for c in fake.commands if c[0] == "uid")


# --- the audit ledger --------------------------------------------------------- #


def test_audit_records_and_lists_newest_first(settings) -> None:
    from assistant.mail import audit

    audit.record(settings, "heartbeat", "archive", "5", "archived: “a”")
    audit.record(settings, "chat:t1", "label", "6", "labeled 'x': “b”")

    rows = audit.recent(settings)
    assert [row["detail"] for row in rows] == ["labeled 'x': “b”", "archived: “a”"]
    assert [row["actor"] for row in rows] == ["chat:t1", "heartbeat"]

    heartbeat_only = audit.recent(settings, actor="heartbeat")
    assert [row["action"] for row in heartbeat_only] == ["archive"]


def test_audit_record_never_raises(settings, monkeypatch) -> None:
    from assistant.mail import audit

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(audit, "_connect", boom)
    audit.record(settings, "heartbeat", "archive", "5", "x")  # logged, not raised


# --- the unread snapshot (per-turn context, no IMAP on the reply path) ------- #


def _fake_messages() -> list[client.Message]:
    return [
        client.Message(uid="1", sender="anna@x.com", subject="Quarterly numbers",
                       date="", unread=True),
        client.Message(uid="2", sender="bob@x.com", subject="", date="", unread=True),
    ]


def test_snapshot_refresh_persists_and_current_serves_it(settings, monkeypatch) -> None:
    from assistant.mail import snapshot

    monkeypatch.setattr(client, "list_recent", lambda s, unread_only=True: _fake_messages())
    assert snapshot.refresh(settings) is not None

    block = snapshot.current(settings)
    assert block.startswith("## Unread mail (snapshot as of ")
    assert "Quarterly numbers" in block and "(no subject)" in block


def test_snapshot_current_never_fetches(settings, monkeypatch) -> None:
    from assistant.mail import snapshot

    monkeypatch.setattr(
        client, "list_recent",
        lambda *a, **k: pytest.fail("current() must never touch the mailbox"),
    )
    assert snapshot.current(settings) == ""  # nothing persisted yet — and no I/O


def test_snapshot_survives_a_restart(settings, monkeypatch) -> None:
    from assistant.mail import snapshot

    monkeypatch.setattr(client, "list_recent", lambda s, unread_only=True: _fake_messages())
    snapshot.refresh(settings)
    # A "restart": nothing in memory, only the persisted file.
    assert "Quarterly numbers" in snapshot.current(settings)


def test_snapshot_maybe_refresh_respects_cadence(settings, monkeypatch) -> None:
    from assistant.mail import snapshot

    calls = {"n": 0}

    def counting(s, unread_only=True):
        calls["n"] += 1
        return _fake_messages()

    monkeypatch.setattr(client, "list_recent", counting)
    snapshot.maybe_refresh(settings)
    snapshot.maybe_refresh(settings)  # fresh — must not refetch
    assert calls["n"] == 1


def test_snapshot_failed_refresh_keeps_previous(settings, monkeypatch) -> None:
    from assistant.mail import snapshot

    monkeypatch.setattr(client, "list_recent", lambda s, unread_only=True: _fake_messages())
    snapshot.refresh(settings)

    def boom(*a, **k):
        raise RuntimeError("mailbox down")

    monkeypatch.setattr(client, "list_recent", boom)
    assert snapshot.refresh(settings) is None  # logged, not raised
    assert "Quarterly numbers" in snapshot.current(settings)  # old snapshot stands


def test_snapshot_invalidate_withholds_until_the_next_tick(settings, monkeypatch) -> None:
    from assistant.mail import snapshot

    calls = {"n": 0}

    def counting(s, unread_only=True):
        calls["n"] += 1
        return _fake_messages()

    monkeypatch.setattr(client, "list_recent", counting)
    snapshot.refresh(settings)
    assert "Quarterly numbers" in snapshot.current(settings)

    snapshot.invalidate(settings)
    assert snapshot.current(settings) == ""  # stale — withheld, no I/O
    snapshot.maybe_refresh(settings)  # the next ticker tick refetches
    assert calls["n"] == 2
    assert "Quarterly numbers" in snapshot.current(settings)


def test_snapshot_invalidate_without_a_snapshot_is_inert(settings) -> None:
    from assistant.mail import snapshot

    snapshot.invalidate(settings)  # nothing persisted — nothing to do
    assert snapshot.current(settings) == ""


def test_snapshot_disabled_is_inert(tmp_path, monkeypatch) -> None:
    from assistant.mail import snapshot

    s = Settings(memory_dir=str(tmp_path / "m"))  # email off
    monkeypatch.setattr(
        client, "list_recent", lambda *a, **k: pytest.fail("disabled must not fetch")
    )
    assert snapshot.refresh(s) is None
    snapshot.maybe_refresh(s)
    assert snapshot.current(s) == ""

    zero = Settings(memory_dir=str(tmp_path / "m2"), enable_email=True,
                    email_snapshot_minutes=0)
    assert snapshot.refresh(zero) is None
    assert snapshot.current(zero) == ""
