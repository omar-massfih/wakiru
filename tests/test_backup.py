"""Drive backup/restore — exercised end-to-end against a fake Drive API.

The single network seam is ``drive._api``; patching it lets the snapshot →
tar.gz → multipart-upload → list → download → extract chain run for real
(including the multipart body formatting) without a token or a socket.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.parse

from assistant.backup import drive
from assistant.config import Settings


def _make_db(path, value: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE kv (v TEXT)")
        conn.execute("INSERT INTO kv (v) VALUES (?)", (value,))
        conn.commit()
    finally:
        conn.close()


def _read_db(path) -> str:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT v FROM kv").fetchone()[0]
    finally:
        conn.close()


def _settings(tmp_path, **overrides) -> Settings:
    base = {"storage_backend": "local", "memory_dir": str(tmp_path), "drive_backup_enabled": True}
    base.update(overrides)
    return Settings(**base)


class FakeDrive:
    """In-memory stand-in for Drive, routed by method + URL like the real API."""

    def __init__(self) -> None:
        self.files: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []
        self._seq = 0

    def _new_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}{self._seq}"

    def seed_archive(self, folder_id: str, name: str, data: bytes) -> str:
        fid = self._new_id("file")
        self.files[fid] = {
            "name": name,
            "parent": folder_id,
            "createdTime": f"2026-07-28T00:00:{self._seq:02d}Z",
            "data": data,
        }
        return fid

    def api(self, method, url, *, settings, body=b"", headers=None):
        self.calls.append((method, url))
        parts = urllib.parse.urlsplit(url)
        qs = urllib.parse.parse_qs(parts.query)

        if method == "GET" and qs.get("alt") == ["media"]:
            fid = parts.path.rsplit("/", 1)[1]
            return 200, {}, self.files[fid]["data"]

        if method == "GET" and "q" in qs:
            q = qs["q"][0]
            if "mimeType='application/vnd.google-apps.folder'" in q:
                folders = [
                    {"id": i, "name": f["name"]}
                    for i, f in self.files.items()
                    if f.get("folder")
                ]
                return 200, {}, json.dumps({"files": folders}).encode()
            folder_id = q.split("'")[1]  # "'<id>' in parents ..."
            archives = [
                {"id": i, "name": f["name"], "createdTime": f["createdTime"]}
                for i, f in self.files.items()
                if f.get("parent") == folder_id
            ]
            archives.sort(key=lambda f: f["createdTime"], reverse=True)
            return 200, {}, json.dumps({"files": archives}).encode()

        if method == "POST" and "/upload/" in parts.path:
            boundary = headers["Content-Type"].split("boundary=")[1]
            meta, data = _parse_multipart(body, boundary)
            self.seed_archive(meta["parents"][0], meta["name"], data)
            return 200, {}, json.dumps({"id": "uploaded"}).encode()

        if method == "POST":  # create folder
            meta = json.loads(body.decode())
            fid = self._new_id("folder")
            self.files[fid] = {"name": meta["name"], "folder": True}
            return 200, {}, json.dumps({"id": fid}).encode()

        if method == "DELETE":
            self.files.pop(parts.path.rsplit("/", 1)[1], None)
            return 204, {}, b""

        raise AssertionError(f"unexpected {method} {url}")


def _parse_multipart(body: bytes, boundary: str) -> tuple[dict, bytes]:
    segments = body.split(b"--" + boundary.encode())
    meta_raw = segments[1].split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
    file_raw = segments[2].split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
    return json.loads(meta_raw), file_raw


def test_snapshot_captures_every_db(tmp_path):
    settings = _settings(tmp_path)
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    _make_db(settings.memory_path / "tasks.db", "alpha")
    _make_db(settings.memory_path / "index.db", "beta")

    archive = drive.snapshot(settings)
    assert archive is not None
    dest = tmp_path / "restored"
    names = drive._extract(archive.read_bytes(), dest)

    assert set(names) == {"tasks.db", "index.db"}
    assert _read_db(dest / "tasks.db") == "alpha"
    assert _read_db(dest / "index.db") == "beta"


def test_snapshot_none_when_no_dbs(tmp_path):
    settings = _settings(tmp_path)
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    assert drive.snapshot(settings) is None


def test_backup_then_restore_roundtrip(tmp_path, monkeypatch):
    fake = FakeDrive()
    monkeypatch.setattr(drive, "_api", fake.api)

    src = _settings(tmp_path / "node-a")
    src.memory_path.mkdir(parents=True, exist_ok=True)
    _make_db(src.memory_path / "expenses.db", "spent")

    drive.run_backup(src)
    # Folder created + archive stored on the fake Drive.
    assert any(f.get("folder") for f in fake.files.values())
    assert any(f.get("name", "").startswith("wakiru-backup-") for f in fake.files.values())

    # A fresh, empty node restores that archive from Drive.
    dst = _settings(tmp_path / "node-b")
    drive.restore_if_empty(dst)
    assert _read_db(dst.memory_path / "expenses.db") == "spent"


def test_restore_noop_when_dbs_present(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("Drive must not be touched when local DBs exist")

    monkeypatch.setattr(drive, "_api", _boom)
    settings = _settings(tmp_path)
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    _make_db(settings.memory_path / "tasks.db", "keep")

    drive.restore_if_empty(settings)  # must not raise / not call _api
    assert _read_db(settings.memory_path / "tasks.db") == "keep"


def test_restore_noop_when_no_archive(tmp_path, monkeypatch):
    fake = FakeDrive()
    monkeypatch.setattr(drive, "_api", fake.api)
    settings = _settings(tmp_path)

    drive.restore_if_empty(settings)  # empty Drive → start fresh, no crash
    assert not list(settings.memory_path.glob("*.db"))


def test_prune_keeps_newest(tmp_path, monkeypatch):
    fake = FakeDrive()
    monkeypatch.setattr(drive, "_api", fake.api)
    settings = _settings(tmp_path, drive_backup_keep=2)
    folder = fake._new_id("folder")
    fake.files[folder] = {"name": "wakiru-backups", "folder": True}
    for n in ("wakiru-backup-1.tar.gz", "wakiru-backup-2.tar.gz", "wakiru-backup-3.tar.gz"):
        fake.seed_archive(folder, n, b"x")

    drive.prune(settings, folder)

    kept = sorted(f["name"] for f in fake.files.values() if f.get("parent") == folder)
    assert kept == ["wakiru-backup-2.tar.gz", "wakiru-backup-3.tar.gz"]
