#!/usr/bin/env python3
"""One-time helper to mint a Google CalDAV OAuth2 *refresh token* for Wakiru.

Run this once on a machine with a browser. It walks the Google consent screen via a
loopback redirect (Google retired the copy-paste "OOB" flow in 2022), exchanges the
resulting code for tokens, and prints the long-lived refresh token you paste into
``.env`` as ``CALDAV_OAUTH_REFRESH_TOKEN``.

    python scripts/caldav_oauth_setup.py \\
        --client-id XXX.apps.googleusercontent.com \\
        --client-secret GOCSPX-... \\
        --email you@gmail.com

Prereqs (Google Cloud Console): the Calendar API enabled, an OAuth **Desktop app**
client, and the consent screen **published** (so the refresh token doesn't expire in
7 days). Stdlib only — no dependencies.
"""

from __future__ import annotations

import argparse
import http.server
import json
import secrets
import socket
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser

_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_SCOPE = "https://www.googleapis.com/auth/calendar"


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    code: str | None = None
    state: str | None = None
    expected_state: str = ""

    def do_GET(self) -> None:
        params = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        _CallbackHandler.code = (params.get("code") or [None])[0]
        _CallbackHandler.state = (params.get("state") or [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        ok = _CallbackHandler.code and _CallbackHandler.state == _CallbackHandler.expected_state
        msg = "Authorized — you can close this tab." if ok else "Authorization failed."
        self.wfile.write(f"<html><body><h3>{msg}</h3></body></html>".encode())

    def log_message(self, *_args) -> None:  # silence the default request logging
        pass


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _exchange(client_id: str, client_secret: str, code: str, redirect_uri: str) -> dict:
    body = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }
    ).encode()
    request = urllib.request.Request(
        _TOKEN_URL, data=body, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-secret", required=True)
    parser.add_argument("--email", required=True, help="the Google account, for the URL hint")
    args = parser.parse_args()

    port = _free_port()
    redirect_uri = f"http://localhost:{port}/"
    state = secrets.token_urlsafe(16)
    _CallbackHandler.expected_state = state

    auth_url = _AUTH_URL + "?" + urllib.parse.urlencode(
        {
            "client_id": args.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": _SCOPE,
            "access_type": "offline",   # ask for a refresh token
            "prompt": "consent",        # force it even on re-auth
            "state": state,
        }
    )

    server = http.server.HTTPServer(("127.0.0.1", port), _CallbackHandler)
    threading.Thread(target=server.handle_request, daemon=True).start()

    print("\nOpening the Google consent screen in your browser…")
    print("If it doesn't open, paste this URL:\n\n" + auth_url + "\n")
    webbrowser.open(auth_url)

    # handle_request serves exactly one request, then the thread ends; wait for it.
    while _CallbackHandler.code is None and _CallbackHandler.state is None:
        threading.Event().wait(0.2)

    if not _CallbackHandler.code or _CallbackHandler.state != state:
        print("Authorization failed (no code, or state mismatch).", file=sys.stderr)
        return 1

    tokens = _exchange(args.client_id, args.client_secret, _CallbackHandler.code, redirect_uri)
    refresh = tokens.get("refresh_token")
    if not refresh:
        print(
            "No refresh_token returned. Re-run after removing this app's access at "
            "https://myaccount.google.com/permissions (Google only returns it on first "
            "consent), and make sure the consent screen is Published.",
            file=sys.stderr,
        )
        return 1

    print("\n=== Add these to .env ===")
    print("CALDAV_AUTH=oauth")
    print(f"CALDAV_URL=https://apps.google.com/calendar/dav/{args.email}/events/")
    print(f"CALDAV_OAUTH_CLIENT_ID={args.client_id}")
    print(f"CALDAV_OAUTH_CLIENT_SECRET={args.client_secret}")
    print(f"CALDAV_OAUTH_REFRESH_TOKEN={refresh}")
    print("ENABLE_CALDAV=true")
    print("ENABLE_CALDAV_WRITE=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
