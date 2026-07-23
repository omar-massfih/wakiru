"""Email tools — read/draft/manage; send only behind the second switch."""
from __future__ import annotations

from ..config import Settings
from ._base import ToolContext, ToolSpec, _int_arg, _params, logger


def _list_email(ctx: ToolContext, unread_only: bool = True) -> str:
    from ..mail import client as mail_client

    messages = mail_client.list_recent(ctx.settings, unread_only=bool(unread_only))
    if not messages:
        return "No messages." if not unread_only else "No unread messages."
    return "\n".join(
        f"- [{m.uid}] {'(unread) ' if m.unread else ''}{m.sender} — {m.subject} ({m.date})"
        for m in messages
    )

def _search_email(
    ctx: ToolContext,
    sender: str = "",
    subject: str = "",
    text: str = "",
    since_days: str = "",
) -> str:
    from ..mail import client as mail_client

    days = _int_arg(since_days, 0)
    if days is None or days < 0:
        return "since_days must be a number of days."
    if not (str(sender).strip() or str(subject).strip() or str(text).strip()):
        return "Give at least one of sender, subject, or text."
    messages = mail_client.search_messages(
        ctx.settings, sender=str(sender), subject=str(subject),
        text=str(text), since_days=days,
    )
    if not messages:
        return "No matching messages."
    return "\n".join(
        f"- [{m.uid}] {m.sender} — {m.subject} ({m.date})" for m in messages
    )

def _read_email(ctx: ToolContext, uid: str) -> str:
    from ..mail import client as mail_client

    message = mail_client.read_message(ctx.settings, str(uid))
    if message is None:
        return f"No message with uid {uid}."
    attachments = (
        f"Attachments: {', '.join(message.attachments)}\n" if message.attachments else ""
    )
    return (
        f"From: {message.sender}\nSubject: {message.subject}\n"
        f"Date: {message.date}\n{attachments}\n{message.body}"
    )

def _draft_email(ctx: ToolContext, to: str, subject: str, body: str, cc: str = "") -> str:
    from ..mail import client as mail_client

    return mail_client.save_draft(ctx.settings, str(to), str(subject), str(body), str(cc))

def _send_email(ctx: ToolContext, to: str, subject: str, body: str, cc: str = "") -> str:
    from ..mail import client as mail_client

    return mail_client.send_message(ctx.settings, str(to), str(subject), str(body), str(cc))

def _ingest_attachment(ctx: ToolContext, uid: str, name: str = "") -> str:
    from ..docs import extract as docs_extract
    from ..docs import store as docs_store
    from ..mail import client as mail_client

    message, fetched = mail_client.read_with_attachment(
        ctx.settings, str(uid), str(name or "")
    )
    if message is None:
        return f"No message with uid {uid}."
    if fetched is None:
        if not message.attachments:
            return "That message has no attachments."
        return (
            "Couldn't pin down one attachment — name one of: "
            + ", ".join(message.attachments)
        )
    filename, content = fetched
    limit = ctx.settings.docs_upload_max_bytes
    if len(content) > limit:
        return f"{filename} exceeds the {limit}-byte ingest limit."
    # Subject in the title keys the dedupe to this message, so re-ingesting the
    # same attachment is refused while a later email with an updated file of
    # the same name is not.
    title = f"{filename} — {message.subject} — email from {message.sender}"
    existing = [
        d for d in docs_store.list_documents(ctx.settings) if d.title == title
    ]
    if existing:
        return (
            f"{filename} from that email is already ingested as document "
            f"{existing[0].id} (“{title}”)."
        )
    try:
        text = docs_extract.extract_text(filename, content)
    except docs_extract.ExtractionError as exc:
        return f"Could not extract text from {filename}: {exc}"
    doc = docs_store.add_document(ctx.settings, title, text)
    return (
        f"Ingested {filename} as document {doc.id} (“{title}”). Its content is "
        "now searchable with search_documents; summarize_document gives an overview."
    )

def _mail_mutated(result: str) -> bool:
    """Whether a mail client result string reports a performed mutation.

    The client returns "No message with uid …" when nothing happened and an
    explanatory "This server has folders…" refusal for unsupported label
    removal; every other return is the summary of a change that was made.
    """
    return not result.startswith(("No message with uid", "This server has"))

def _record_mail_action(
    ctx: ToolContext, action: str, uid: str, detail: str, *, invalidate: bool = False
) -> None:
    """Audit a performed mailbox mutation; optionally stale the unread snapshot."""
    from ..mail import audit as mail_audit

    actor = f"chat:{ctx.thread_id}" if ctx.thread_id else "heartbeat"
    mail_audit.record(ctx.settings, actor, action, uid, detail)
    if invalidate:
        try:
            from ..mail import snapshot as mail_snapshot

            mail_snapshot.invalidate(ctx.settings)
        except Exception:
            logger.debug("mail snapshot invalidation failed", exc_info=True)

def _reply_email(ctx: ToolContext, uid: str, body: str, reply_all: bool = False) -> str:
    from ..mail import client as mail_client

    result = mail_client.save_reply_draft(
        ctx.settings, str(uid), str(body), bool(reply_all)
    )
    if _mail_mutated(result):
        _record_mail_action(ctx, "reply_draft", str(uid), result)
    return result

def _send_reply(ctx: ToolContext, uid: str, body: str, reply_all: bool = False) -> str:
    from ..mail import client as mail_client

    result = mail_client.send_reply(ctx.settings, str(uid), str(body), bool(reply_all))
    if _mail_mutated(result):
        _record_mail_action(ctx, "reply_sent", str(uid), result)
    return result

def _archive_email(ctx: ToolContext, uid: str) -> str:
    from ..mail import client as mail_client

    result = mail_client.archive_message(ctx.settings, str(uid))
    if _mail_mutated(result):
        _record_mail_action(ctx, "archive", str(uid), result, invalidate=True)
    return result

def _mark_email_read(ctx: ToolContext, uid: str, unread: bool = False) -> str:
    from ..mail import client as mail_client

    result = mail_client.mark_read(ctx.settings, str(uid), bool(unread))
    if _mail_mutated(result):
        _record_mail_action(ctx, "mark_read", str(uid), result, invalidate=True)
    return result

def _label_email(ctx: ToolContext, uid: str, label: str, remove: bool = False) -> str:
    from ..mail import client as mail_client

    result = mail_client.set_label(ctx.settings, str(uid), str(label), bool(remove))
    if _mail_mutated(result):
        _record_mail_action(ctx, "label", str(uid), result)
    return result

def _email_tools(settings: Settings) -> list[ToolSpec]:
    tools = [
        ToolSpec(
            "list_email",
            "List recent mailbox messages (never marks anything read).",
            _params(
                {"unread_only": ("boolean", "Only unread messages (default true)")},
                [],
            ),
            _list_email,
        ),
        ToolSpec(
            "search_email",
            "Search the whole inbox server-side, old mail included.",
            _params(
                {
                    "sender": ("string", "Match the From header"),
                    "subject": ("string", "Match the Subject header"),
                    "text": ("string", "Match anywhere in the message"),
                    "since_days": ("string", "Only the last N days"),
                },
                [],
            ),
            _search_email,
        ),
        ToolSpec(
            "read_email",
            "Read one message's body by uid.",
            _params({"uid": ("string", "Message uid from list_email")}, ["uid"]),
            _read_email,
        ),
        ToolSpec(
            "draft_email",
            "Save an email draft to the drafts folder (does not send).",
            _params(
                {
                    "to": ("string", "Recipient address"),
                    "subject": ("string", "Subject line"),
                    "body": ("string", "Plain-text body"),
                    "cc": ("string", "Optional Cc address(es), comma-separated"),
                },
                ["to", "subject", "body"],
            ),
            _draft_email,
        ),
        ToolSpec(
            "reply_email",
            "Draft a properly threaded reply to a message by uid (saves to the "
            "drafts folder; does not send). Prefer this over draft_email when "
            "answering an existing message.",
            _params(
                {
                    "uid": ("string", "Message uid from list_email"),
                    "body": ("string", "Plain-text reply body"),
                    "reply_all": (
                        "boolean",
                        "Also Cc the original To/Cc recipients (default false)",
                    ),
                },
                ["uid", "body"],
            ),
            _reply_email,
        ),
        ToolSpec(
            "archive_email",
            "Archive a message: remove it from the inbox without deleting it "
            "(recoverable — on Gmail it stays in All Mail).",
            _params({"uid": ("string", "Message uid from list_email")}, ["uid"]),
            _archive_email,
        ),
        ToolSpec(
            "mark_email_read",
            "Mark a message read (or back to unread with unread=true). Reading "
            "with read_email never does this implicitly.",
            _params(
                {
                    "uid": ("string", "Message uid from list_email"),
                    "unread": ("boolean", "Mark unread instead (default false)"),
                },
                ["uid"],
            ),
            _mark_email_read,
        ),
        ToolSpec(
            "label_email",
            "Apply or remove a label on a message (Gmail); on folder-based "
            "servers, labeling moves the message to that folder.",
            _params(
                {
                    "uid": ("string", "Message uid from list_email"),
                    "label": ("string", "Label or folder name"),
                    "remove": ("boolean", "Remove the label instead (default false)"),
                },
                ["uid", "label"],
            ),
            _label_email,
        ),
    ]
    if settings.enable_docs:
        tools.append(
            ToolSpec(
                "ingest_attachment",
                "Ingest an email attachment (PDF, DOCX, or text-like) into the "
                "user's documents so it becomes searchable and summarizable. "
                "Never marks the message read.",
                _params(
                    {
                        "uid": ("string", "Message uid from list_email"),
                        "name": (
                            "string",
                            "Attachment filename (needed only when the message"
                            " has several)",
                        ),
                    },
                    ["uid"],
                ),
                _ingest_attachment,
            )
        )
    if settings.enable_email_send:
        tools.append(
            ToolSpec(
                "send_email",
                "Send an email. Only after the user explicitly confirmed sending "
                "this exact message in this conversation.",
                _params(
                    {
                        "to": ("string", "Recipient address"),
                        "subject": ("string", "Subject line"),
                        "body": ("string", "Plain-text body"),
                        "cc": ("string", "Optional Cc address(es), comma-separated"),
                    },
                    ["to", "subject", "body"],
                ),
                _send_email,
            )
        )
        tools.append(
            ToolSpec(
                "send_reply",
                "Send a threaded reply to a message by uid. Only after the user "
                "explicitly confirmed sending this exact reply in this "
                "conversation.",
                _params(
                    {
                        "uid": ("string", "Message uid from list_email"),
                        "body": ("string", "Plain-text reply body"),
                        "reply_all": (
                            "boolean",
                            "Also Cc the original To/Cc recipients (default false)",
                        ),
                    },
                    ["uid", "body"],
                ),
                _send_reply,
            )
        )
    return tools
