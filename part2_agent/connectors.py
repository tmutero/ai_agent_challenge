"""
Tool connectors behind small interfaces, each with two implementations:

    Calendar   GoogleCalendar  | FixtureCalendar
    Mail       Gmail           | FixtureMail
    Web        HttpWeb (real HTTP scrape + real DNS MX lookups)

The agent only sees the interface, so the same orchestration runs against a
real Google account (live demo) or the scenario fixtures (offline tests, and a
fallback if OAuth fails mid-interview). Both are seeded from ``scenario.py``.
"""

from __future__ import annotations

import base64
import email.utils
import os
import re
from dataclasses import dataclass, field
from email.message import EmailMessage
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from . import gmail_query, scenario

ROOT = Path(__file__).parent.parent
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",   # create the summary draft
    "https://www.googleapis.com/auth/gmail.insert",    # seed demo emails only
]
TOKEN_PATH = ROOT / "token.json"
CREDENTIALS_PATH = ROOT / "credentials.json"


# --- interfaces -------------------------------------------------------------------

class Calendar(Protocol):
    def search_events(self, query: str, time_min: str, time_max: str) -> list[dict]: ...
    def get_event(self, event_id: str) -> dict: ...


class Mail(Protocol):
    def search(self, query: str, max_results: int = 20) -> list[dict]: ...
    def get(self, message_id: str) -> dict: ...
    def create_draft(self, to: str, subject: str, body: str) -> str: ...


# --- fixtures -----------------------------------------------------------------------

class FixtureCalendar:
    def __init__(self):
        self.events = [scenario.EVENT, *scenario.OTHER_EVENTS]

    def search_events(self, query: str, time_min: str, time_max: str) -> list[dict]:
        # Google Calendar's q= is free text over summary/location/description.
        in_range = [e for e in self.events if time_min[:10] <= e["start"][:10] <= time_max[:10]]
        docs = [{"text": f"{e['summary']} {e.get('location', '')} {e.get('description', '')}".lower()} for e in in_range]
        return [_event_summary(in_range[i]) for i in gmail_query.search(query or "", docs)]

    def get_event(self, event_id: str) -> dict:
        for e in self.events:
            if e["id"] == event_id:
                return e
        raise KeyError(f"No event {event_id}")


class FixtureMail:
    def __init__(self):
        self.messages = {m["id"]: m for m in scenario.EMAILS}
        self.drafts: list[dict] = []

    def search(self, query: str, max_results: int = 20) -> list[dict]:
        msgs = list(self.messages.values())
        hits = gmail_query.search(query, [gmail_query.message_doc(m) for m in msgs])
        return [_message_summary(msgs[i]) for i in hits[:max_results]]

    def get(self, message_id: str) -> dict:
        return self.messages[message_id]

    def create_draft(self, to: str, subject: str, body: str) -> str:
        self.drafts.append({"to": to, "subject": subject, "body": body})
        return f"fixture-draft-{len(self.drafts)}"


def _event_summary(e: dict) -> dict:
    return {"id": e["id"], "summary": e["summary"], "start": e["start"], "location": e.get("location", ""),
            "attendee_count": len(e.get("attendees", []))}


def _message_summary(m: dict) -> dict:
    return {"id": m["id"], "from": m["from"], "subject": m["subject"], "date": m["date"],
            "snippet": m["body"][:120].replace("\n", " ")}


# --- Google -----------------------------------------------------------------------

def google_credentials():
    """Load (and refresh) the OAuth token created by ``python -m part2_agent.google_setup auth``.
    On a hosted deploy the token JSON can be supplied as the GOOGLE_TOKEN_JSON secret."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    from common.llm import secret

    raw = secret("GOOGLE_TOKEN_JSON")
    if raw:
        import json
        creds = Credentials.from_authorized_user_info(json.loads(raw), SCOPES)
    elif TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    else:
        raise RuntimeError("No Google token. Run: python -m part2_agent.google_setup auth")
    if not creds.valid and creds.refresh_token:
        creds.refresh(Request())
        if not raw:
            TOKEN_PATH.write_text(creds.to_json())
    return creds


class GoogleCalendar:
    def __init__(self, creds=None):
        from googleapiclient.discovery import build
        self.svc = build("calendar", "v3", credentials=creds or google_credentials(), cache_discovery=False)

    def search_events(self, query: str, time_min: str, time_max: str) -> list[dict]:
        res = self.svc.events().list(calendarId="primary", q=query or None, timeMin=time_min, timeMax=time_max,
                                     singleEvents=True, orderBy="startTime", maxResults=25).execute()
        return [_event_summary(self._normalise(e)) for e in res.get("items", [])]

    def get_event(self, event_id: str) -> dict:
        return self._normalise(self.svc.events().get(calendarId="primary", eventId=event_id).execute())

    @staticmethod
    def _normalise(e: dict) -> dict:
        return {"id": e["id"], "summary": e.get("summary", ""), "location": e.get("location", ""),
                "description": e.get("description", ""),
                "start": e["start"].get("dateTime", e["start"].get("date")),
                "end": e["end"].get("dateTime", e["end"].get("date")),
                "attendees": e.get("attendees", [])}


class Gmail:
    def __init__(self, creds=None):
        from googleapiclient.discovery import build
        self.svc = build("gmail", "v1", credentials=creds or google_credentials(), cache_discovery=False)

    def search(self, query: str, max_results: int = 20) -> list[dict]:
        res = self.svc.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
        return [_message_summary(self.get(m["id"])) for m in res.get("messages", [])]

    def get(self, message_id: str) -> dict:
        m = self.svc.users().messages().get(userId="me", id=message_id, format="full").execute()
        headers = {h["name"].lower(): h["value"] for h in m["payload"].get("headers", [])}
        return {"id": m["id"], "from": headers.get("from", ""), "to": headers.get("to", ""),
                "date": headers.get("date", ""), "subject": headers.get("subject", ""),
                "body": _plain_body(m["payload"])}

    def create_draft(self, to: str, subject: str, body: str) -> str:
        msg = EmailMessage()
        msg["To"], msg["Subject"] = to, subject
        msg.set_content(body)
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        draft = self.svc.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
        return draft["id"]


def _plain_body(payload: dict) -> str:
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", "replace")
    for part in payload.get("parts", []) or []:
        text = _plain_body(part)
        if text:
            return text
    return ""


# --- Web ----------------------------------------------------------------------------

@dataclass
class DirectoryFetch:
    url: str
    firms: list[dict]
    people: list[dict]
    fetched_via: str            # "http" | "local-file-fallback"
    notes: list[str] = field(default_factory=list)


class HttpWeb:
    """Real web access: scrape the public partner directory over HTTP and resolve
    MX records over DNS. The directory URL is configurable; if it is unreachable
    the bundled copy is parsed instead and the fallback is recorded."""

    def __init__(self, directory_url: str | None = None):
        from common.llm import secret
        self.directory_url = directory_url or secret(
            "DIRECTORY_URL", "http://localhost:8501/app/static/partner_directory.html")
        self._cache: DirectoryFetch | None = None

    def directory(self) -> DirectoryFetch:
        if self._cache:
            return self._cache
        import httpx
        notes = []
        try:
            resp = httpx.get(self.directory_url, timeout=10, follow_redirects=True)
            resp.raise_for_status()
            html, via = resp.text, "http"
        except Exception as exc:  # noqa: BLE001
            notes.append(f"HTTP fetch of {self.directory_url} failed ({type(exc).__name__}); parsed bundled copy")
            html, via = scenario.DIRECTORY_FILE.read_text(encoding="utf-8"), "local-file-fallback"
        firms, people = parse_directory(html)
        if via == "http" and not people:   # e.g. a login/redirect page served with HTTP 200
            notes.append(f"{self.directory_url} returned no directory tables; parsed bundled copy")
            firms, people = parse_directory(scenario.DIRECTORY_FILE.read_text(encoding="utf-8"))
            via = "local-file-fallback"
        self._cache = DirectoryFetch(self.directory_url, firms, people, via, notes)
        return self._cache

    @staticmethod
    def mx_lookup(domain: str) -> dict:
        return _mx_lookup(domain.lower())


def parse_directory(html: str) -> tuple[list[dict], list[dict]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    def rows(table_id: str, cols: list[str]) -> list[dict]:
        table = soup.find("table", id=table_id)
        out = []
        for tr in table.find("tbody").find_all("tr") if table else []:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            out.append(dict(zip(cols, cells)))
        return out

    return rows("firms", ["firm", "domain", "fsp"]), rows("representatives", ["name", "title", "firm", "email", "phone"])


@lru_cache(maxsize=256)
def _mx_lookup(domain: str) -> dict:
    import dns.exception
    import dns.resolver

    try:
        answers = dns.resolver.resolve(domain, "MX", lifetime=5)
        hosts = sorted(str(r.exchange).rstrip(".") for r in answers)
        if hosts == [""]:
            return {"domain": domain, "status": "null_mx", "detail": "domain explicitly accepts no mail"}
        return {"domain": domain, "status": "ok", "mx": hosts[:3]}
    except dns.resolver.NXDOMAIN:
        return {"domain": domain, "status": "nxdomain", "detail": "domain does not exist"}
    except (dns.resolver.NoAnswer, dns.resolver.NoNameservers):
        return {"domain": domain, "status": "no_mx", "detail": "no mail servers for domain"}
    except dns.exception.DNSException as exc:
        return {"domain": domain, "status": "error", "detail": type(exc).__name__}


# --- factory --------------------------------------------------------------------------

@dataclass
class Toolset:
    calendar: Calendar
    mail: Mail
    web: HttpWeb
    mode: str


def make_toolset(mode: str | None = None) -> Toolset:
    mode = mode or os.environ.get("AGENT_DATA_MODE", "fixture")
    if mode == "google":
        creds = google_credentials()
        return Toolset(GoogleCalendar(creds), Gmail(creds), HttpWeb(), "google")
    return Toolset(FixtureCalendar(), FixtureMail(), HttpWeb(), "fixture")


def email_date(raw: str) -> str:
    """RFC 2822 date header -> ISO 8601 (for ordering RSVP statements)."""
    try:
        return email.utils.parsedate_to_datetime(raw).isoformat()
    except (TypeError, ValueError):
        return raw
