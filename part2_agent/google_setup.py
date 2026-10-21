"""
Google account setup for the live Part 2 demo.

    python -m part2_agent.google_setup auth    # browser OAuth consent -> token.json
    python -m part2_agent.google_setup seed    # write the demo event + emails into the account
    python -m part2_agent.google_setup reset   # delete the demo calendar events again

The hosted app calls ``ensure_seeded()`` before a live run, so production needs no manual seed step.

Use a personal/throwaway Google account: Workspace admins often block unverified
OAuth apps. Seeding is side-effect free for third parties: calendar invites are
created with sendUpdates="none" (no notifications), and emails are *inserted*
into the mailbox (users.messages.insert), never sent. Firm domains are reserved
``.example`` addresses.
"""

from __future__ import annotations

import base64
import sys
from datetime import datetime, timezone
from email.message import EmailMessage

from . import scenario
from .connectors import CREDENTIALS_PATH, SCOPES, TOKEN_PATH, google_credentials

DEMO_TAG = {"private": {"kestrelDemo": "1"}}


def auth() -> None:
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not CREDENTIALS_PATH.exists():
        sys.exit(f"Missing {CREDENTIALS_PATH}. Download your OAuth 'Desktop app' client JSON there first.")
    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent")
    TOKEN_PATH.write_text(creds.to_json())
    print(f"Saved {TOKEN_PATH}")


def _event_body(e: dict) -> dict:
    attendees = [{k: v for k, v in a.items() if k in ("email", "displayName", "responseStatus")}
                 for a in e.get("attendees", []) if not (a.get("organizer") or a.get("self") or a.get("resource"))]
    return {"summary": e["summary"], "location": e.get("location", ""), "description": e.get("description", ""),
            "start": {"dateTime": e["start"]}, "end": {"dateTime": e["end"]},
            "attendees": attendees, "extendedProperties": DEMO_TAG,
            "guestsCanSeeOtherGuests": False, "reminders": {"useDefault": False}}


def _services():
    from googleapiclient.discovery import build

    creds = google_credentials()
    return (build("calendar", "v3", credentials=creds, cache_discovery=False),
            build("gmail", "v1", credentials=creds, cache_discovery=False))


def is_seeded(cal=None) -> bool:
    """True while the demo event is still upcoming. Scenario dates are relative to the seeding day,
    so once the seeded event has passed the account needs a fresh seed."""
    cal = cal or _services()[0]
    items = cal.events().list(calendarId="primary", privateExtendedProperty="kestrelDemo=1",
                              timeMin=datetime.now(timezone.utc).isoformat(), singleEvents=True,
                              maxResults=50).execute().get("items", [])
    return any(e.get("summary") == scenario.EVENT["summary"] for e in items)


def ensure_seeded(log=print) -> bool:
    """Seed the demo data unless an upcoming demo event already exists. Returns True if it seeded."""
    cal, gmail = _services()
    if is_seeded(cal):
        return False
    reset(log, cal)   # clear stale (past) demo events before re-seeding
    seed(log, cal, gmail)
    return True


def seed(log=print, cal=None, gmail=None) -> None:
    if cal is None or gmail is None:
        cal, gmail = _services()
    me = gmail.users().getProfile(userId="me").execute()["emailAddress"]

    for e in [scenario.EVENT, *scenario.OTHER_EVENTS]:
        created = cal.events().insert(calendarId="primary", body=_event_body(e), sendUpdates="none").execute()
        log(f"event  {created['id']}  {e['summary']}")

    for m in scenario.EMAILS:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Date"], msg["Subject"] = m["from"], me, m["date"], m["subject"]
        msg["X-Kestrel-Demo"] = "1"
        msg.set_content(m["body"])
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        res = gmail.users().messages().insert(
            userId="me", internalDateSource="dateHeader",
            body={"raw": raw, "labelIds": ["INBOX", "UNREAD"]}).execute()
        log(f"email  {res['id']}  {m['subject']}")
    log("Seeded. The organiser of the event is your own account.")


def reset(log=print, cal=None) -> None:
    cal = cal or _services()[0]
    items = cal.events().list(calendarId="primary", privateExtendedProperty="kestrelDemo=1",
                              maxResults=250).execute().get("items", [])
    for e in items:
        cal.events().delete(calendarId="primary", eventId=e["id"], sendUpdates="none").execute()
        log(f"deleted event {e['summary']}")
    log("Calendar reset. To remove the demo emails, search Gmail for "
        "'subject:(roadshow OR factsheets)' and delete them (this script has no delete scope by design).")


if __name__ == "__main__":
    {"auth": auth, "seed": seed, "reset": reset}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
