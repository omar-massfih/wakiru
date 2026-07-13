"""SSRF guard tests — DNS and HTTP are faked, nothing touches the network."""

from __future__ import annotations

import socket

import pytest

from assistant import netguard


def _addrinfo(ip: str) -> list:
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 80))]


def test_rejects_non_http_schemes() -> None:
    with pytest.raises(netguard.BlockedURLError, match="http"):
        netguard.require_public_url("file:///etc/passwd")
    with pytest.raises(netguard.BlockedURLError, match="http"):
        netguard.require_public_url("ftp://example.com/x")


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",  # loopback
        "10.0.0.8",  # RFC 1918
        "192.168.1.5",  # RFC 1918
        "169.254.169.254",  # link-local / cloud metadata
        "100.64.0.1",  # CGNAT
        "::1",  # IPv6 loopback
    ],
)
def test_rejects_non_public_addresses(monkeypatch: pytest.MonkeyPatch, ip: str) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo(ip))
    with pytest.raises(netguard.BlockedURLError, match="non-public"):
        netguard.require_public_url("http://host.example.com/")


def test_accepts_public_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34"))
    netguard.require_public_url("https://example.com/page")  # must not raise


def test_unresolvable_host_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a, **k):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(netguard.BlockedURLError, match="resolve"):
        netguard.require_public_url("http://ghost.invalid/")


def test_redirect_to_private_host_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = {"public.example.com": "93.184.216.34", "internal.example.com": "10.0.0.8"}
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda host, *a, **k: _addrinfo(resolved[host])
    )

    def fake_open(request, timeout):
        raise netguard._RedirectSignal("http://internal.example.com/steal")

    monkeypatch.setattr(netguard, "_open", fake_open)
    with pytest.raises(netguard.BlockedURLError, match="non-public"):
        netguard.urlopen_public("http://public.example.com/", timeout=5)


def test_too_many_redirects_are_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34"))

    def fake_open(request, timeout):
        raise netguard._RedirectSignal("http://elsewhere.example.com/")

    monkeypatch.setattr(netguard, "_open", fake_open)
    with pytest.raises(netguard.BlockedURLError, match="redirect"):
        netguard.urlopen_public("http://start.example.com/", timeout=5, max_redirects=3)
