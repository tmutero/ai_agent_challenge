"""Part 2: deterministic rules, Gmail query evaluation, and consolidation -- all offline."""

from datetime import timedelta

import pytest

from part2_agent import cleansing as C
from part2_agent import gmail_query, scenario
from part2_agent.connectors import FixtureCalendar, FixtureMail, HttpWeb
from part2_agent.extraction import ExtractedPerson, ExtractionResult, ground
from part2_agent.workspace import Workspace


@pytest.mark.parametrize("raw,expected", [
    ("082 555 0143", "+27825550143"),
    ("+27 (0)83 555 0177", "+27835550177"),
    ("(021) 555-0198", "+27215550198"),
    ("+27 21 555 0110", "+27215550110"),
])
def test_phone_normalisation(raw, expected):
    assert C.normalize_phone(raw) == (expected, None)


def test_truncated_sa_phone_rejected():
    e164, problem = C.normalize_phone("082 555 01")
    assert e164 is None and "9 digits" in problem


def test_names():
    assert C.normalize_name("LERATO MOKOENA") == "Lerato Mokoena"
    assert C.normalize_name("sarah VAN DER merwe") == "Sarah van der Merwe"
    assert C.normalize_name("michael o'connor") == "Michael O'Connor"
    assert C.name_from_email("J.Botha@KarooCapital.example") == "J Botha"
    assert C.same_person("J Botha", "Johan Botha")[0]
    assert C.same_person("Michael Oconnor", "Michael O'Connor")[0]
    assert not C.same_person("Johan Botha", "Kabelo Dube")[0]


def test_domain_typos_and_company_canonicalisation():
    known = {"highveldfo.example", "gmail.com"}
    assert C.domain_typo_of("highveldfo.exmaple", known) == "highveldfo.example"
    assert C.domain_typo_of("gmial.com", known) == "gmail.com"
    assert C.domain_typo_of("gmail.com", known) is None
    firms = [f["firm"] for f in scenario.FIRMS]
    assert C.canonical_company("Fynbos Wealth", firms)[0] == "Fynbos Wealth Partners"
    assert C.canonical_company("Karoo Capital (Pty) Ltd", firms)[0] == "Karoo Capital"


@pytest.mark.parametrize("query,expected", [
    ('"Cape Town" roadshow', {"msg-001", "msg-002", "msg-003", "msg-004", "msg-005"}),
    ("subject:(registration OR registrations OR rsvp)", {"msg-004", "msg-005"}),
    (f"roadshow -subject:apologies before:{scenario.TODAY - timedelta(days=4):%Y/%m/%d}", {"msg-001", "msg-002"}),
    ("from:(nomsa OR thabo)", {"msg-004", "msg-005"}),
    ("factsheets", {"msg-006"}),
])
def test_fixture_mail_understands_gmail_syntax(query, expected):
    assert {m["id"] for m in FixtureMail().search(query)} == expected


def test_parenthesised_or_group():
    docs = [{"text": t} for t in ("cpt roadshow", "cape town roadshow", "cape town dinner", "cpt")]
    assert gmail_query.search('("Cape Town" OR CPT) roadshow', docs) == [0, 1]


def test_grounding_rejects_values_not_in_email():
    text = "Johan Botha\nCell: 082 555 01"
    rec, rejected = ground(ExtractedPerson(full_name="Johan Botha", email="johan@made.up", phone="082 555 0166",
                                           company=None, job_title=None, attendance="attending", evidence=""), text)
    assert rec["email"] is None and rec["phone"] is None and len(rejected) == 2


# --- consolidation with hand-written "extractions" (no LLM) ------------------------

def _extraction(msg_id, people):
    return ExtractionResult(msg_id, True, [
        {"full_name": n, "email": e, "phone": p, "company": c, "job_title": t, "attendance": a, "evidence": ""}
        for n, e, p, c, t, a in people])


@pytest.fixture
def workspace():
    ws = Workspace(HttpWeb(directory_url="http://127.0.0.1:9/unreachable"))   # forces bundled-copy fallback
    mail = FixtureMail()
    ws.add_calendar_event(FixtureCalendar().get_event("evt-roadshow-cpt"))
    data = {
        "msg-001": [("Lerato Mokoena", "lerato.mokoena@fynboswealth.example", "082 555 0143", "Fynbos Wealth Partners", "Senior Financial Planner", "attending")],
        "msg-002": [("Johan Botha", "johan.botha@karoocapital.example", "082 555 01", "Karoo Capital", "Portfolio Manager", "attending"),
                    ("Kabelo Dube", None, None, None, None, "requested_by_colleague")],
        "msg-003": [("Ayanda Zulu", "ayanda.zulu@tablebayfa.example", "+27 21 555 0110", "Table Bay Financial Advisers", None, "declined")],
        "msg-004": [("LERATO MOKOENA", "LERATO.MOKOENA@FYNBOSWEALTH.EXAMPLE", "0825550143", "Fynbos Wealth Partners", None, "attending"),
                    ("Sarah van der Merwe", "sarah.vdmerwe@fynboswealth.example", "+27 (0)83 555 0177", "Fynbos Wealth", "Assoc.", "attending"),
                    ("Kabelo Dube", "kabelo.dube@karoocapital.example", "084 555 0102", "Karoo Capital", "Analyst", "attending"),
                    ("Nomsa", "nomsa.events@kestrel-am.example", None, None, None, "unknown")],
        "msg-005": [("Thabo Nkosi", "thabo.nkosi.demo@gmail.com", None, None, "Independent Financial Adviser", "attending")],
    }
    for mid, people in data.items():
        ws.add_extraction(mail.get(mid), _extraction(mid, people))
    return ws


def _row(rows, name):
    return next(r for r in rows if r["name"] == name)


def test_consolidation_dedups_and_excludes(workspace):
    rows, _ = workspace.table()
    names = sorted(r["name"] for r in rows)
    assert names == ["Ayanda Zulu", "Grace Adeyemi", "Johan Botha", "Kabelo Dube", "Lerato Mokoena",
                     "Michael O'Connor", "Priya Naidoo", "Sarah van der Merwe", "Thabo Nkosi"]
    excluded = {e["email"] for e in workspace.excluded + workspace.internal_excluded}
    assert {"events@kestrel-am.example", "westin-ballroom-b@resource.kestrel-am.example",
            "nomsa.events@kestrel-am.example"} <= excluded


def test_field_resolution_and_flags(workspace):
    rows, issues = workspace.table()
    johan = _row(rows, "Johan Botha")
    assert johan["email"] == "johan.botha@karoocapital.example" and johan["phone"] == "+27825550166"
    assert "j.botha@karoocapital.example" in johan["alt_emails"]
    assert "invite_address_mismatch" in johan["flags"]
    assert _row(rows, "Ayanda Zulu")["rsvp"] == "declined"
    assert _row(rows, "Kabelo Dube")["on_invite"] is False
    sarah = _row(rows, "Sarah van der Merwe")
    assert sarah["company"] == "Fynbos Wealth Partners" and sarah["title"] == "Associate Financial Planner"
    assert _row(rows, "Michael O'Connor")["status"] == "verified"
    codes = {i.issue_id: i for i in issues}
    priya = codes["priya.naidoo@highveldfo.exmaple:email_domain_typo"]
    assert priya.severity == "error" and priya.suggestion.startswith("priya.naidoo@highveldfo.example")
    assert "grace.adeyemi.demo@gmial.com:email_domain_typo" in codes
    assert "thabo.nkosi.demo@gmail.com:missing_phone" in codes


def test_correction_requires_evidence_and_is_revalidated(workspace):
    key = "priya.naidoo@highveldfo.exmaple"
    bad = workspace.propose_correction(key, "email", "priya@highveldfo.example", "directory", "guess")
    assert not bad.accepted
    dns = workspace.propose_correction("grace.adeyemi.demo@gmial.com", "email", "grace.adeyemi.demo@gmail.com",
                                       "dns:gmail.com", "typo")
    assert not dns.accepted and "mailbox" in dns.check
    good = workspace.propose_correction(key, "email", "priya.naidoo@highveldfo.example", "directory", "directory")
    assert good.accepted
    rows, issues = workspace.table()
    priya = _row(rows, "Priya Naidoo")
    assert priya["email"] == "priya.naidoo@highveldfo.example"
    ids = {i.issue_id for i in issues}
    assert f"{key}:email_domain_typo" not in ids
    assert f"{key}:invite_address_mismatch" in ids     # the invite still points at the typo


def test_build_is_deterministic(workspace):
    assert workspace.table()[0] == workspace.table()[0]


def test_scenario_is_always_in_the_future():
    """The demo must not go stale: the event is ahead of 'today', emails are behind it."""
    from part2_agent.connectors import email_date
    assert scenario.EVENT["start"][:10] > scenario.TODAY.isoformat()
    assert all(email_date(m["date"])[:10] < scenario.TODAY.isoformat() for m in scenario.EMAILS)
    assert str(scenario.EVENT_DAY.day) in scenario.EMAILS[2]["subject"]
