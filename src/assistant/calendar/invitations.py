"""Provider-neutral invitation identity and response helpers.

Calendar providers describe the current user differently: Google marks an
attendee as ``self``, while CalDAV normally identifies them by address.  Keep
the conservative resolution rules here so agenda rendering and writes cannot
disagree about which attendee is safe to change.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from . import store

RESPONSES = frozenset({"accepted", "tentative", "declined"})


@dataclass(frozen=True)
class Invitation:
    """The classification of an event from the configured user's perspective."""

    self_index: int | None
    attendee: dict | None
    reason: str
    pending: bool = False

    @property
    def is_invitation(self) -> bool:
        return self.reason == "invitation"


def _identity_emails(settings: Settings) -> set[str]:
    values = (
        settings.caldav_username,
        settings.email_address,
        None
        if str(settings.google_calendar_id or "").lower() == "primary"
        else settings.google_calendar_id,
    )
    identities: set[str] = set()
    for value in values:
        email = store.normalize_email(value)
        if email.count("@") == 1 and all(email.split("@", 1)):
            identities.add(email)
    return identities


def _self_index(settings: Settings, attendees: list[dict]) -> tuple[int | None, str]:
    marked = [index for index, attendee in enumerate(attendees) if attendee.get("self") is True]
    if len(marked) > 1:
        return None, "ambiguous self attendee"
    if len(marked) == 1:
        return marked[0], ""

    identities = _identity_emails(settings)
    matched = [
        index for index, attendee in enumerate(attendees)
        if store.normalize_email(attendee.get("email")) in identities
    ]
    if len(matched) > 1:
        return None, "ambiguous self attendee"
    if not matched:
        return None, "self attendee not found"
    return matched[0], ""


def _pending(attendee: dict) -> bool:
    status = "".join(
        character for character in str(attendee.get("status") or "").lower()
        if character.isalnum()
    )
    return status == "needsaction" or (not status and attendee.get("rsvp") is True)


def classify(settings: Settings, event: store.Event) -> Invitation:
    """Classify ``event`` without guessing when the user's identity is uncertain."""
    attendees = store.load_attendees(event)
    index, error = _self_index(settings, attendees)
    if error:
        return Invitation(None, None, error)
    assert index is not None
    attendee = attendees[index]
    organizer = store.load_organizer(event)
    if organizer.get("self") is True or (
        organizer.get("email")
        and store.normalize_email(organizer.get("email")) == attendee["email"]
    ):
        return Invitation(index, attendee, "not an invitation")
    return Invitation(index, attendee, "invitation", pending=_pending(attendee))


def updated_attendees(
    settings: Settings, event: store.Event, response: str
) -> tuple[list[dict] | None, str | None]:
    """Copy attendees and change only the uniquely identified user's status."""
    if response not in RESPONSES:
        return None, "invalid response"
    invitation = classify(settings, event)
    if not invitation.is_invitation:
        return None, invitation.reason
    assert invitation.self_index is not None
    attendees = [dict(attendee) for attendee in store.load_attendees(event)]
    attendees[invitation.self_index]["status"] = response
    return attendees, None
