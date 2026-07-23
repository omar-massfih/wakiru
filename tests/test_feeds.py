"""Feed-fetcher tests — RSS/Atom parsing, the cache, and netguard wrapping."""

from __future__ import annotations

import pytest

from assistant import feeds

_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Example Blog</title>
  <item><title>Post two</title><link>https://x.test/2</link></item>
  <item><title>Post one</title><link>https://x.test/1</link></item>
</channel></rss>"""

_ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Example Feed</title>
  <entry><title>Alpha</title><link href="https://a.test/alpha"/></entry>
  <entry><title>Beta</title><link href="https://a.test/beta"/></entry>
</feed>"""


def test_parse_rss_items_in_order() -> None:
    entries = feeds.parse_entries(_RSS)
    assert [(e.title, e.link) for e in entries] == [
        ("Post two", "https://x.test/2"),
        ("Post one", "https://x.test/1"),
    ]


def test_parse_atom_entries_with_href_links() -> None:
    entries = feeds.parse_entries(_ATOM)
    assert [(e.title, e.link) for e in entries] == [
        ("Alpha", "https://a.test/alpha"),
        ("Beta", "https://a.test/beta"),
    ]


def test_parse_rejects_non_xml() -> None:
    with pytest.raises(feeds.FeedError):
        feeds.parse_entries("<html>not a feed")


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self, n: int) -> bytes:
        return self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def test_fetch_caches_until_forced(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_open(url, *, timeout, headers=None, max_redirects=5):
        calls["n"] += 1
        return _Response(_RSS.encode())

    monkeypatch.setattr(feeds, "urlopen_public", fake_open)
    monkeypatch.setattr(feeds, "_cache", {})
    url = "https://x.test/feed.xml"
    assert len(feeds.fetch_entries(url)) == 2
    assert len(feeds.fetch_entries(url)) == 2
    assert calls["n"] == 1  # second read came from the cache
    feeds.fetch_entries(url, force=True)
    assert calls["n"] == 2


def test_blocked_url_becomes_feed_error(monkeypatch) -> None:
    def blocked(url, *, timeout, headers=None, max_redirects=5):
        raise feeds.BlockedURLError("non-public address")

    monkeypatch.setattr(feeds, "urlopen_public", blocked)
    monkeypatch.setattr(feeds, "_cache", {})
    with pytest.raises(feeds.FeedError):
        feeds.fetch_entries("http://10.0.0.1/feed")
