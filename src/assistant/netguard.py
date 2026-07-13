"""SSRF guard for server-side fetches of user- or feed-supplied URLs.

Both outbound fetchers (document URL ingest, ICS feed sync) go through
:func:`urlopen_public`, which refuses anything that is not a public http(s)
host and re-validates every redirect hop, so a public URL can't bounce the
server into cloud metadata (169.254.169.254), localhost, or the LAN.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse
import urllib.request

_DEFAULT_MAX_REDIRECTS = 5


class BlockedURLError(ValueError):
    """The URL was refused by the guard (bad scheme or non-public address)."""


def require_public_url(url: str) -> None:
    """Raise :class:`BlockedURLError` unless ``url`` is http(s) on a public host.

    Resolves the hostname and refuses any address that is not globally
    routable (loopback, RFC 1918, link-local/cloud-metadata, CGNAT,
    reserved, …). The check and the later connection are separate lookups,
    so this is defense-in-depth, not a hard guarantee against DNS rebinding.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise BlockedURLError("only http(s) URLs are allowed")
    host = parsed.hostname
    if not host:
        raise BlockedURLError("URL has no host")
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise BlockedURLError(f"could not resolve host {host!r}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise BlockedURLError(
                f"host {host!r} resolves to non-public address {address}"
            )


class _RedirectSignal(Exception):
    """Internal: a redirect the caller must validate before following."""

    def __init__(self, target: str) -> None:
        self.target = target


class _StopRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise _RedirectSignal(newurl)


def _open(request: urllib.request.Request, timeout: float):
    """One non-redirecting HTTP open — the seam tests replace."""
    return urllib.request.build_opener(_StopRedirects).open(request, timeout=timeout)


def urlopen_public(
    url: str,
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
    max_redirects: int = _DEFAULT_MAX_REDIRECTS,
):
    """Open ``url``, following redirects manually and re-validating each hop."""
    for _ in range(max_redirects + 1):
        require_public_url(url)
        request = urllib.request.Request(url, headers=headers or {})
        try:
            return _open(request, timeout)
        except _RedirectSignal as signal:
            url = urllib.parse.urljoin(url, signal.target)
    raise BlockedURLError(f"more than {max_redirects} redirects")
