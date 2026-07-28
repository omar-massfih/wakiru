"""One-way backup of the local SQLite stores to Google Drive.

The local backend keeps everything in ~19 SQLite ``.db`` files under the memory
directory, on a single node's disk. That is fast and dependency-free but not
durable, so this module makes it durable: on a cadence (and on shutdown) it takes
a *consistent* snapshot of every DB and uploads one tar.gz to a dedicated Drive
folder; on a fresh / empty memory dir at startup it restores the newest archive.
Drive is backup only — the local SQLite files stay the single source of truth, so
there is no two-way sync and no conflict handling to get wrong.

Snapshots use SQLite's online-backup API (``Connection.backup``), which copies a
live, WAL-mode database into a clean single file without stopping writers and —
unlike ``VACUUM INTO`` — without needing the sqlite-vec extension loaded to carry
its ``vec0`` shadow tables. Every network call is stdlib ``urllib`` with a Bearer
token, SSRF-guarded through :mod:`assistant.netguard`, mirroring the calendar and
mail REST clients. All entry points are best-effort: a failure is logged and
never propagates into the daemon's loops or its startup.
"""

from __future__ import annotations

import io
import json
import logging
import secrets
import shutil
import sqlite3
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from .. import netguard
from ..config import Settings

logger = logging.getLogger(__name__)

_FILES_URL = "https://www.googleapis.com/drive/v3/files"
_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"
_FOLDER_MIME = "application/vnd.google-apps.folder"
_ARCHIVE_MIME = "application/gzip"
_ARCHIVE_PREFIX = "wakiru-backup-"
# Uploads/downloads are tens of MB, so allow far longer than a REST round-trip.
_TIMEOUT_SECONDS = 300
_MAX_REDIRECTS = 5


class DriveError(RuntimeError):
    """A Drive API call returned an unexpected status or failed in transport."""


# --- Snapshot / archive -----------------------------------------------------


def _copy_db(src: Path, dst: Path) -> None:
    """Copy a live SQLite DB to ``dst`` as a clean single file (online backup)."""
    source = sqlite3.connect(src)
    try:
        target = sqlite3.connect(dst)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()


def snapshot(settings: Settings) -> Path | None:
    """Write a tar.gz of the whole memory dir into a fresh temp dir.

    Every ``*.db`` is copied consistently via the online-backup API; every other
    on-disk file is copied verbatim, preserving its relative path — crucially the
    note markdown tree (``<kind>/<name>.md`` + ``MEMORY.md``), which is the
    *source of truth* for note bodies (``index.db`` only holds the derived vector
    index). Excludes WAL sidecars and the ``*_token.json`` OAuth caches (secrets,
    regenerated from the refresh token). Returns the archive path (caller cleans
    up its parent), or ``None`` when there is nothing to back up yet.
    """
    memory = settings.memory_path
    dbs = sorted(memory.glob("*.db"))
    extras = sorted(
        p
        for p in memory.rglob("*")
        if p.is_file()
        and p.suffix != ".db"
        and not p.name.endswith((".db-wal", ".db-shm", ".tmp"))
        and not p.name.endswith("_token.json")
    )
    if not dbs and not extras:
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    workdir = Path(tempfile.mkdtemp(prefix="wakiru-snap-"))
    staging = workdir / "mem"
    staging.mkdir()
    for db in dbs:
        _copy_db(db, staging / db.name)
    for src in extras:
        dst = staging / src.relative_to(memory)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    archive = workdir / f"{_ARCHIVE_PREFIX}{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staging, arcname=".")
    shutil.rmtree(staging, ignore_errors=True)
    return archive


def _extract(data: bytes, dest: Path) -> list[str]:
    """Extract a memory archive into ``dest``, preserving subdirs; refuse any
    absolute path or ``..`` traversal."""
    dest.mkdir(parents=True, exist_ok=True)
    restored: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name[2:] if member.name.startswith("./") else member.name
            if not name or name.startswith("/") or ".." in name.split("/"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(handle.read())
            restored.append(name)
    return restored


# --- Drive REST -------------------------------------------------------------


def _lower_headers(headers) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in (headers.items() if headers else [])}


def _api(
    method: str,
    url: str,
    *,
    settings: Settings,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """One Drive round-trip → ``(status, lowercased_headers, body)``. THE test seam.

    Bearer auth, SSRF-guarded with per-hop redirect re-validation, never raises on
    a 4xx/5xx (the status is returned so callers map it); only transport failure
    raises.
    """
    from .drive_oauth import access_token

    hdrs = {"Authorization": f"Bearer {access_token(settings)}", **(headers or {})}
    target = url
    for _ in range(_MAX_REDIRECTS + 1):
        netguard.require_public_url(target)
        request = urllib.request.Request(target, data=body or None, method=method, headers=hdrs)
        opener = urllib.request.build_opener(netguard._StopRedirects)
        try:
            response = opener.open(request, timeout=_TIMEOUT_SECONDS)
        except netguard._RedirectSignal as signal:
            target = urllib.parse.urljoin(target, signal.target)
            continue
        except urllib.error.HTTPError as exc:
            return exc.code, _lower_headers(exc.headers), exc.read()
        except OSError as exc:
            raise DriveError(f"Drive {method} {target} failed: {exc}") from exc
        with response:
            status = getattr(response, "status", 0) or response.getcode()
            return status, _lower_headers(response.headers), response.read()
    raise DriveError(f"too many redirects for {url}")


def _folder_id(settings: Settings, *, create: bool) -> str | None:
    """Resolve the backup folder's Drive id, optionally creating it."""
    name = settings.drive_backup_folder.replace("'", r"\'")
    query = f"mimeType='{_FOLDER_MIME}' and name='{name}' and trashed=false"
    url = _FILES_URL + "?" + urllib.parse.urlencode(
        {"q": query, "fields": "files(id,name)", "spaces": "drive"}
    )
    status, _, body = _api("GET", url, settings=settings)
    if status != 200:
        raise DriveError(f"listing the backup folder failed: {status} {body[:200]!r}")
    files = json.loads(body or b"{}").get("files", [])
    if files:
        return str(files[0]["id"])
    if not create:
        return None
    meta = json.dumps(
        {"name": settings.drive_backup_folder, "mimeType": _FOLDER_MIME}
    ).encode()
    status, _, body = _api(
        "POST",
        _FILES_URL + "?fields=id",
        settings=settings,
        body=meta,
        headers={"Content-Type": "application/json; charset=UTF-8"},
    )
    if status not in (200, 201):
        raise DriveError(f"creating the backup folder failed: {status} {body[:200]!r}")
    return str(json.loads(body)["id"])


def _list_archives(settings: Settings, folder_id: str) -> list[dict]:
    """Backup archives in the folder, newest first."""
    query = f"'{folder_id}' in parents and trashed=false"
    url = _FILES_URL + "?" + urllib.parse.urlencode(
        {
            "q": query,
            "fields": "files(id,name,createdTime)",
            "orderBy": "createdTime desc",
            "pageSize": "1000",
        }
    )
    status, _, body = _api("GET", url, settings=settings)
    if status != 200:
        raise DriveError(f"listing archives failed: {status} {body[:200]!r}")
    files = json.loads(body or b"{}").get("files", [])
    return [f for f in files if str(f.get("name", "")).startswith(_ARCHIVE_PREFIX)]


def upload(settings: Settings, archive: Path, folder_id: str) -> None:
    """Multipart-upload ``archive`` into the backup folder."""
    data = archive.read_bytes()
    boundary = "wakiru-" + secrets.token_hex(16)
    meta = json.dumps({"name": archive.name, "parents": [folder_id]}).encode()
    payload = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b"Content-Type: application/json; charset=UTF-8\r\n\r\n",
            meta,
            b"\r\n",
            f"--{boundary}\r\n".encode(),
            f"Content-Type: {_ARCHIVE_MIME}\r\n\r\n".encode(),
            data,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    url = _UPLOAD_URL + "?" + urllib.parse.urlencode({"uploadType": "multipart", "fields": "id"})
    status, _, body = _api(
        "POST",
        url,
        settings=settings,
        body=payload,
        headers={"Content-Type": f"multipart/related; boundary={boundary}"},
    )
    if status not in (200, 201):
        raise DriveError(f"uploading {archive.name} failed: {status} {body[:200]!r}")


def download(settings: Settings, file_id: str) -> bytes:
    url = _FILES_URL + "/" + urllib.parse.quote(file_id, safe="") + "?alt=media"
    status, _, body = _api("GET", url, settings=settings)
    if status != 200:
        raise DriveError(f"downloading {file_id} failed: {status} {body[:200]!r}")
    return body


def prune(settings: Settings, folder_id: str) -> None:
    """Delete all but the newest ``drive_backup_keep`` archives."""
    for stale in _list_archives(settings, folder_id)[settings.drive_backup_keep :]:
        url = _FILES_URL + "/" + urllib.parse.quote(str(stale["id"]), safe="")
        status, _, _ = _api("DELETE", url, settings=settings)
        if status not in (200, 204):
            logger.warning(
                "backup: could not delete old archive %s (%s)", stale.get("name"), status
            )


# --- Entry points -----------------------------------------------------------


def run_backup(settings: Settings) -> None:
    """Snapshot every SQLite DB and upload it to Drive, pruning old archives.

    Best-effort: any failure is logged and swallowed so a backup never disturbs
    the loop that scheduled it.
    """
    if not settings.drive_backup_enabled:
        return
    try:
        archive = snapshot(settings)
    except Exception:
        logger.exception("backup: snapshot failed")
        return
    if archive is None:
        return
    try:
        folder_id = _folder_id(settings, create=True)
        assert folder_id is not None  # create=True never returns None
        upload(settings, archive, folder_id)
        prune(settings, folder_id)
        logger.info("backup: uploaded %s to Drive", archive.name)
    except Exception:
        logger.exception("backup: upload to Drive failed")
    finally:
        shutil.rmtree(archive.parent, ignore_errors=True)


def restore_if_empty(settings: Settings) -> None:
    """Restore the newest Drive archive when the memory dir has no SQLite DBs.

    Runs once at startup before anything opens a store. A no-op when local DBs
    already exist. Best-effort: on any failure the daemon just starts fresh.
    """
    if not settings.drive_backup_enabled:
        return
    memory = settings.memory_path
    if any(memory.glob("*.db")):
        return
    try:
        folder_id = _folder_id(settings, create=False)
        if folder_id is None:
            logger.info("backup: no Drive backup folder yet; starting fresh")
            return
        archives = _list_archives(settings, folder_id)
        if not archives:
            logger.info("backup: no archive on Drive yet; starting fresh")
            return
        latest = archives[0]
        data = download(settings, str(latest["id"]))
        restored = _extract(data, memory)
        logger.info("backup: restored %d DBs from %s", len(restored), latest["name"])
    except Exception:
        logger.exception("backup: restore from Drive failed; starting fresh")
