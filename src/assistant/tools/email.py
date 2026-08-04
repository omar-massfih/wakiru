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

def _recipient_entries(value: object, field: str) -> tuple[list[str] | None, str | None]:
    """Accept the array schema plus legacy single/address-list strings."""
    if isinstance(value, list):
        return list(value), None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return [], None
        if "," in text:
            from ..mail import client as mail_client

            try:
                canonical = mail_client._require_address_list(text, field=field)
            except ValueError:
                pass  # A People name may itself contain a comma.
            else:
                return canonical.split(", "), None
        return [text], None
    return None, f"{field} recipients must be a list of People names or email addresses."


def _resolve_recipients(
    ctx: ToolContext, value: object, field: str, *, required: bool
) -> tuple[list[str] | None, str | None]:
    """Resolve a complete recipient field without performing any mail writes."""
    from ..mail import client as mail_client
    from ..people import store as people_store

    entries, error = _recipient_entries(value, field)
    if error:
        return None, error
    if required and not entries:
        return None, f"{field} must contain at least one People name or email address."

    resolved: dict[str, None] = {}
    for raw in entries or []:
        if not isinstance(raw, str) or not raw.strip():
            return None, (
                f"Every {field} recipient must be a non-empty People name or email address."
            )
        text = raw.strip()
        try:
            canonical = mail_client._require_address_list(text, field=field)
        except ValueError:
            canonical = ""
        if canonical and "," not in canonical:
            email = canonical.lower()
        else:
            if "@" in text or text.lower().startswith("mailto:"):
                return None, f"Invalid {field} email address: {text}."
            matches = people_store.find_people(ctx.settings, text)
            if not matches:
                return None, (
                    f'No Person matches {field} recipient "{text}". Use a stored '
                    "People name or a direct email address."
                )
            if len(matches) > 1:
                candidates = ", ".join(
                    f'{person.name} ({person.id}, {person.email or "no email"})'
                    for person in matches[:5]
                )
                more = f", +{len(matches) - 5} more" if len(matches) > 5 else ""
                return None, (
                    f'Ambiguous {field} recipient "{text}" — matches: '
                    f"{candidates}{more}. Use a more specific People name, exact "
                    "Person id, or direct email."
                )
            person = matches[0]
            if not person.email:
                return None, (
                    f'Person "{person.name}" ({person.id}) has no email. Add one '
                    "to the Person or use a direct email address."
                )
            try:
                email = mail_client._require_address_list(person.email, field=field).lower()
            except ValueError:
                return None, (
                    f'Person "{person.name}" ({person.id}) has an invalid email. '
                    "Update the Person or use a direct email address."
                )
            if "," in email:
                return None, (
                    f'Person "{person.name}" ({person.id}) has an invalid email. '
                    "Update the Person or use a direct email address."
                )
        resolved.setdefault(email, None)
    return list(resolved), None


def _resolved_email_fields(
    ctx: ToolContext, to: object, cc: object
) -> tuple[str | None, str | None, str | None]:
    """Resolve both fields atomically; To wins over Cc for duplicate addresses."""
    resolved_to, error = _resolve_recipients(ctx, to, "To", required=True)
    if error:
        return None, None, error
    resolved_cc, error = _resolve_recipients(ctx, cc, "Cc", required=False)
    if error:
        return None, None, error
    to_addresses = resolved_to or []
    to_set = set(to_addresses)
    cc_addresses = [address for address in (resolved_cc or []) if address not in to_set]
    return ", ".join(to_addresses), ", ".join(cc_addresses), None


def _draft_email(
    ctx: ToolContext, to: object, subject: str, body: str, cc: object = ""
) -> str:
    from ..mail import client as mail_client

    resolved_to, resolved_cc, error = _resolved_email_fields(ctx, to, cc)
    if error:
        return error
    return mail_client.save_draft(
        ctx.settings, resolved_to or "", str(subject), str(body), resolved_cc or ""
    )

def _send_email(
    ctx: ToolContext, to: object, subject: str, body: str, cc: object = ""
) -> str:
    from ..mail import client as mail_client

    resolved_to, resolved_cc, error = _resolved_email_fields(ctx, to, cc)
    if error:
        return error
    return mail_client.send_message(
        ctx.settings, resolved_to or "", str(subject), str(body), resolved_cc or ""
    )


def _email_recipient_params() -> dict:
    schema = _params(
        {
            "subject": ("string", "Subject line"),
            "body": ("string", "Plain-text body"),
        },
        ["to", "subject", "body"],
    )
    item_description = "Direct email address, People name, or exact Person ID"
    schema["properties"]["to"] = {
        "type": "array",
        "items": {"type": "string"},
        "description": f"One or more recipients; each item is a {item_description}",
    }
    schema["properties"]["cc"] = {
        "type": "array",
        "items": {"type": "string"},
        "description": f"Optional Cc recipients; each item is a {item_description}",
    }
    return schema

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
            _email_recipient_params(),
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
                chat_only=True,
            )
        )
    if settings.enable_email_send:
        tools.append(
            ToolSpec(
                "send_email",
                "Send an email. Only after the user explicitly confirmed sending "
                "this exact message in this conversation.",
                _email_recipient_params(),
                _send_email,
                chat_only=True,
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
                chat_only=True,
            )
        )
    return tools
