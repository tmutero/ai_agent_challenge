"""
The Part 2 demo scenario: one source of truth for

  * the offline fixture adapters (FixtureCalendar / FixtureMail), and
  * ``google_setup seed``, which writes the *same* event and emails into a real
    Google account, so the live demo and the offline tests see identical data.

All people and firms are synthetic. Firm domains use the RFC 2606 reserved
``.example`` TLD, so nothing here can ever reach a real mailbox.

Deliberate data-quality problems (what the agent must catch):
  - Lerato Mokoena appears twice (calendar + registration list in UPPER CASE)
  - Johan Botha is on the invite as J.Botha@KarooCapital.example (no display name)
    but emails from johan.botha@... and his signature phone is truncated
  - Ayanda Zulu is "tentative" in the calendar but later emails to decline
  - Priya Naidoo's invite address has a typo'd TLD (.exmaple)
  - Michael O'Connor's display name is lower-case, phone in (021) 555-0198 format
  - Sarah van der Merwe's mobile has a "+27 (0)" trunk prefix; title is abbreviated
  - Kabelo Dube registered by email but is NOT on the calendar invite
  - Thabo Nkosi uses a personal Gmail; no company or phone anywhere
  - Grace Adeyemi's invite address is on "gmial.com" (typo of gmail.com)
  - A meeting-room resource and the organiser are on the invite (must be excluded)
  - An unrelated newsletter is in the mailbox (must be ignored)
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

# Dates are relative to today so the demo never goes stale: the event is ~3 weeks
# out and the emails arrived over the past week. SCENARIO_TODAY pins it for tests.
TODAY = date.fromisoformat(os.environ["SCENARIO_TODAY"]) if os.environ.get("SCENARIO_TODAY") else date.today()
SAST = timezone(timedelta(hours=2))
EVENT_DAY = TODAY + timedelta(days=22)


def _iso(day: date, hh: int, mm: int = 0) -> str:
    return datetime.combine(day, time(hh, mm), SAST).isoformat()


def _mail_date(days_ago: int, hh: int, mm: int) -> str:
    return format_datetime(datetime.combine(TODAY - timedelta(days=days_ago), time(hh, mm), SAST))


DAY_TEXT = f"{EVENT_DAY.day} {EVENT_DAY:%B}"          # e.g. "12 November"
ORDINAL = f"{EVENT_DAY.day}{'th' if 11 <= EVENT_DAY.day % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(EVENT_DAY.day % 10, 'th')}"

ORGANISER = "events@kestrel-am.example"

EVENT = {
    "id": "evt-roadshow-cpt",
    "summary": "Q4 Investor Roadshow - Cape Town",
    "location": "The Westin Cape Town, Convention Square",
    "description": "Kestrel Q4 market outlook and fund update for partner advisers. Breakfast served from 08:30.",
    "start": _iso(EVENT_DAY, 9),
    "end": _iso(EVENT_DAY, 12),
    "attendees": [
        {"email": ORGANISER, "displayName": "Kestrel Events", "organizer": True, "self": True, "responseStatus": "accepted"},
        {"email": "lerato.mokoena@fynboswealth.example", "displayName": "Lerato Mokoena", "responseStatus": "accepted"},
        {"email": "J.Botha@KarooCapital.example", "responseStatus": "accepted"},
        {"email": "ayanda.zulu@tablebayfa.example", "displayName": "Ayanda Zulu", "responseStatus": "tentative"},
        {"email": "priya.naidoo@highveldfo.exmaple", "displayName": "Priya Naidoo", "responseStatus": "needsAction"},
        {"email": "michael.oconnor@umoyarc.example", "displayName": "michael oconnor", "responseStatus": "accepted"},
        {"email": "sarah.vdmerwe@fynboswealth.example", "displayName": "Sarah van der Merwe", "responseStatus": "accepted"},
        {"email": "thabo.nkosi.demo@gmail.com", "displayName": "Thabo Nkosi", "responseStatus": "needsAction"},
        {"email": "grace.adeyemi.demo@gmial.com", "displayName": "Grace Adeyemi", "responseStatus": "needsAction"},
        {"email": "westin-ballroom-b@resource.kestrel-am.example", "displayName": "Westin Ballroom B", "resource": True,
         "responseStatus": "accepted"},
    ],
}

OTHER_EVENTS = [
    {"id": "evt-ic", "summary": "Investment Committee", "start": _iso(EVENT_DAY - timedelta(days=7), 14),
     "end": _iso(EVENT_DAY - timedelta(days=7), 16), "location": "Boardroom", "description": "", "attendees": []},
    {"id": "evt-roadshow-jhb", "summary": "Q4 Investor Roadshow - Johannesburg", "start": _iso(EVENT_DAY + timedelta(days=7), 9),
     "end": _iso(EVENT_DAY + timedelta(days=7), 12), "location": "Sandton Convention Centre", "description": "", "attendees": [
         {"email": ORGANISER, "organizer": True, "self": True, "responseStatus": "accepted"}]},
]

EMAILS = [
    {
        "id": "msg-001",
        "from": "Lerato Mokoena <lerato.mokoena@fynboswealth.example>",
        "to": ORGANISER,
        "date": _mail_date(7, 8, 12),
        "subject": "Re: Q4 Investor Roadshow - Cape Town",
        "body": f"""Hi Kestrel team,

Thanks for the invite - I'll definitely be there on the {ORDINAL}. Vegetarian breakfast please.

Kind regards
Lerato Mokoena
Senior Financial Planner | Fynbos Wealth Partners
M: 082 555 0143
""",
    },
    {
        "id": "msg-002",
        "from": "Johan Botha <johan.botha@karoocapital.example>",
        "to": ORGANISER,
        "date": _mail_date(6, 10, 40),
        "subject": "Roadshow attendance - Karoo Capital",
        "body": """Morning,

Confirming I'll attend the Cape Town roadshow. My colleague Kabelo Dube would also like to join - can you add him?

Johan Botha
Portfolio Manager, Karoo Capital
Cell: 082 555 01
""",
    },
    {
        "id": "msg-003",
        "from": "Ayanda Zulu <ayanda.zulu@tablebayfa.example>",
        "to": ORGANISER,
        "date": _mail_date(2, 16, 5),
        "subject": f"Apologies - Roadshow {EVENT_DAY.day} {EVENT_DAY:%b}",
        "body": f"""Hi all,

Unfortunately I can no longer attend the Cape Town roadshow on {DAY_TEXT} due to a client commitment.
Please send me the slides afterwards.

Regards,
Ayanda Zulu
Table Bay Financial Advisers
Tel: +27 21 555 0110
""",
    },
    {
        "id": "msg-004",
        "from": "Nomsa (Events) <nomsa.events@kestrel-am.example>",
        "to": ORGANISER,
        "date": _mail_date(1, 9, 30),
        "subject": "FW: Roadshow registrations (Cape Town)",
        "body": """Hi team, registrations received via the web form so far - please reconcile with the invite list.

Name                 | Company                        | Email                               | Mobile             | Title
LERATO MOKOENA       | Fynbos Wealth Partners         | LERATO.MOKOENA@FYNBOSWEALTH.EXAMPLE | 0825550143         | Senior Financial Planner
Sarah van der Merwe  | Fynbos Wealth                  | sarah.vdmerwe@fynboswealth.example  | +27 (0)83 555 0177 | Assoc.
Kabelo Dube          | Karoo Capital                  | kabelo.dube@karoocapital.example    | 084 555 0102       | Analyst
Michael O'Connor     | Umoya Retirement Consultants   | michael.oconnor@umoyarc.example     | (021) 555-0198     | Director

Thanks,
Nomsa
""",
    },
    {
        "id": "msg-005",
        "from": "Thabo Nkosi <thabo.nkosi.demo@gmail.com>",
        "to": ORGANISER,
        "date": _mail_date(1, 12, 14),
        "subject": "Roadshow RSVP",
        "body": """Hi, count me in for the Cape Town roadshow.

Thabo Nkosi
Independent Financial Adviser
""",
    },
    {
        "id": "msg-006",
        "from": "Kestrel Marketing <marketing@kestrel-am.example>",
        "to": ORGANISER,
        "date": _mail_date(5, 7, 0),
        "subject": "Monthly fund factsheets - October",
        "body": "The October factsheets for all Kestrel funds are now available on the adviser portal.",
    },
]

# --- Public partner directory (the web reference source) -------------------------

FIRMS = [
    {"firm": "Fynbos Wealth Partners", "domain": "fynboswealth.example", "fsp": "FSP 40211"},
    {"firm": "Karoo Capital", "domain": "karoocapital.example", "fsp": "FSP 38870"},
    {"firm": "Table Bay Financial Advisers", "domain": "tablebayfa.example", "fsp": "FSP 45102"},
    {"firm": "Highveld Family Office", "domain": "highveldfo.example", "fsp": "FSP 51934"},
    {"firm": "Umoya Retirement Consultants", "domain": "umoyarc.example", "fsp": "FSP 47766"},
]

DIRECTORY = [
    {"name": "Lerato Mokoena", "title": "Senior Financial Planner", "firm": "Fynbos Wealth Partners",
     "email": "lerato.mokoena@fynboswealth.example", "phone": "+27 82 555 0143"},
    {"name": "Sarah van der Merwe", "title": "Associate Financial Planner", "firm": "Fynbos Wealth Partners",
     "email": "sarah.vdmerwe@fynboswealth.example", "phone": "+27 83 555 0177"},
    {"name": "Johan Botha", "title": "Portfolio Manager", "firm": "Karoo Capital",
     "email": "johan.botha@karoocapital.example", "phone": "+27 82 555 0166"},
    {"name": "Kabelo Dube", "title": "Investment Analyst", "firm": "Karoo Capital",
     "email": "kabelo.dube@karoocapital.example", "phone": "+27 84 555 0102"},
    {"name": "Ayanda Zulu", "title": "Financial Adviser", "firm": "Table Bay Financial Advisers",
     "email": "ayanda.zulu@tablebayfa.example", "phone": "+27 21 555 0110"},
    {"name": "Priya Naidoo", "title": "Chief Investment Officer", "firm": "Highveld Family Office",
     "email": "priya.naidoo@highveldfo.example", "phone": "+27 11 555 0187"},
    {"name": "Michael O'Connor", "title": "Director", "firm": "Umoya Retirement Consultants",
     "email": "michael.oconnor@umoyarc.example", "phone": "+27 21 555 0198"},
]

STATIC_DIR = Path(__file__).parent.parent / "static"
DIRECTORY_FILE = STATIC_DIR / "partner_directory.html"


def directory_html() -> str:
    firm_rows = "\n".join(
        f'<tr><td class="firm">{f["firm"]}</td><td class="domain">{f["domain"]}</td><td class="fsp">{f["fsp"]}</td></tr>'
        for f in FIRMS)
    people_rows = "\n".join(
        f'<tr><td class="name">{p["name"]}</td><td class="title">{p["title"]}</td><td class="firm">{p["firm"]}</td>'
        f'<td class="email">{p["email"]}</td><td class="phone">{p["phone"]}</td></tr>' for p in DIRECTORY)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Kestrel Partner Directory</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;color:#1f2933}}table{{border-collapse:collapse;margin-bottom:2rem}}
td,th{{border:1px solid #cbd2d9;padding:.4rem .7rem;text-align:left}}th{{background:#1f3b57;color:#fff}}</style></head>
<body>
<h1>Kestrel Partner Directory</h1>
<p>Public register of accredited partner firms and their representatives (synthetic demo data).</p>
<h2>Accredited firms</h2>
<table id="firms"><thead><tr><th>Firm</th><th>Email domain</th><th>FSP number</th></tr></thead><tbody>
{firm_rows}
</tbody></table>
<h2>Representatives</h2>
<table id="representatives"><thead><tr><th>Name</th><th>Title</th><th>Firm</th><th>Email</th><th>Phone</th></tr></thead><tbody>
{people_rows}
</tbody></table>
</body></html>
"""


def write_directory() -> None:
    STATIC_DIR.mkdir(exist_ok=True)
    DIRECTORY_FILE.write_text(directory_html(), encoding="utf-8")


if __name__ == "__main__":
    write_directory()
    print(f"wrote {DIRECTORY_FILE}")
