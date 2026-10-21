"""
The agent's working memory and the deterministic consolidation engine.

  observations (raw, with provenance)  ->  build()  ->  attendees + issues
                                             ^
                    corrections (agent-proposed, evidence-checked)

``build()`` is a pure function of (observations, directory, DNS results,
corrections): calling it twice gives the same table. The agent never edits the
table directly; it can only add observations (by reading sources) or propose a
correction that cites evidence, which is checked before it is accepted. After a
correction the row is *re-validated* by the same rules, so a bad fix shows up as
a new issue rather than silently passing.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import asdict, dataclass, field

from . import cleansing as C
from .connectors import DirectoryFetch, HttpWeb, email_date

FIELDS = ("name", "email", "phone", "company", "title")


@dataclass
class Observation:
    obs_id: str
    source: str              # calendar | email
    ref: str                 # event id / message id
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    company: str | None = None
    title: str | None = None
    attendance: str | None = None
    at: str | None = None    # when the statement was made (emails); None = calendar baseline
    evidence: str = ""
    name_inferred: bool = False


@dataclass
class Issue:
    issue_id: str
    person: str
    field: str
    severity: str            # error | warning | info
    code: str
    message: str
    suggestion: str | None = None
    status: str = "open"     # open | human_review
    review_note: str | None = None


@dataclass
class Correction:
    corr_id: str
    person: str
    field: str
    old: str | None
    new: str
    evidence: str
    rationale: str
    accepted: bool
    check: str


@dataclass
class Attendee:
    key: str
    name: str | None
    email: str | None
    phone: str | None
    company: str | None
    title: str | None
    rsvp: str
    on_invite: bool
    directory_match: str | None           # "email" | "name" | None
    alt_emails: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)   # field -> provenance
    notes: list[str] = field(default_factory=list)          # merges, conflicts, ignored values
    observation_ids: list[str] = field(default_factory=list)


class Workspace:
    def __init__(self, web: HttpWeb):
        self.web = web
        self.event: dict | None = None
        self.observations: list[Observation] = []
        self.excluded: list[dict] = []
        self.messages: dict[str, dict] = {}
        self.extraction_rejections: list[str] = []
        self.irrelevant_messages: list[str] = []
        self.corrections: list[Correction] = []
        self.human_review: dict[str, str] = {}
        self.internal_excluded: list[dict] = []
        self._ids = itertools.count(1)

    # --- ingest ---------------------------------------------------------------------

    def add_calendar_event(self, event: dict) -> dict:
        self.event = event
        self.observations = [o for o in self.observations if o.source != "calendar"]
        self.excluded = []
        kept = []
        for a in event.get("attendees", []):
            if a.get("organizer") or a.get("self") or a.get("resource"):
                why = "organiser/self" if (a.get("organizer") or a.get("self")) else "room/resource"
                self.excluded.append({"email": a["email"], "reason": why})
                continue
            name, inferred = a.get("displayName"), False
            if not name:
                name, inferred = C.name_from_email(a["email"]), True
            self.observations.append(Observation(
                f"O{next(self._ids)}", "calendar", event["id"], name=name, email=a["email"],
                attendance=C.RSVP_FROM_CALENDAR.get(a.get("responseStatus", "needsAction"), "no response"),
                evidence=f"invite attendee {a['email']} ({a.get('responseStatus')})", name_inferred=inferred))
            kept.append({"email": a["email"], "displayName": a.get("displayName"), "responseStatus": a.get("responseStatus")})
        return {"attendees": kept, "excluded": self.excluded}

    def add_extraction(self, message: dict, result) -> list[str]:
        self.messages[message["id"]] = message
        self.observations = [o for o in self.observations if o.ref != message["id"]]
        self.extraction_rejections += result.rejected
        if not result.relevant:
            self.irrelevant_messages.append(message["id"])
            return []
        ids = []
        for p in result.people:
            oid = f"O{next(self._ids)}"
            self.observations.append(Observation(
                oid, "email", message["id"], name=p.get("full_name"), email=p.get("email"),
                phone=p.get("phone"), company=p.get("company"), title=p.get("job_title"),
                attendance=p.get("attendance"), at=email_date(message["date"]), evidence=p.get("evidence", "")))
            ids.append(oid)
        return ids

    # --- consolidation -----------------------------------------------------------------

    def _internal_domains(self) -> set[str]:
        """The organisation's own domains: INTERNAL_DOMAINS plus the organiser's
        domain -- unless that is a free-mail domain (a personal Gmail organiser
        must not exclude every Gmail attendee)."""
        from common.llm import secret
        domains = {d.strip().lower() for d in (secret("INTERNAL_DOMAINS", "kestrel-am.example") or "").split(",") if d.strip()}
        for a in (self.event or {}).get("attendees", []):
            dom = C.email_domain(C.normalize_email(a["email"]))
            if a.get("organizer") and dom and dom not in C.FREE_MAIL:
                domains.add(dom)
        return domains

    def _external(self) -> list[Observation]:
        """Drop people on the organisation's own domains (staff who forwarded or
        sent lists are senders, not attendees) and record why."""
        internal = self._internal_domains()
        is_internal = [C.email_domain(C.normalize_email(o.email)) in internal for o in self.observations]
        self.internal_excluded = [
            {"email": o.email, "reason": "internal staff (sender/forwarder), not an attendee", "source": f"{o.source}:{o.ref}"}
            for o, inside in zip(self.observations, is_internal) if inside]
        return [o for o, inside in zip(self.observations, is_internal) if not inside]

    def _group(self) -> list[tuple[list[Observation], list[str]]]:
        """Entity resolution: same normalised email, else same person by name with a
        compatible email domain / company. Returns groups with merge notes."""
        groups: list[tuple[list[Observation], list[str]]] = []
        ordered = sorted(self._external(), key=lambda o: (o.source != "calendar", o.at or ""))
        for o in ordered:
            em = C.normalize_email(o.email)
            target = None
            for g, notes in groups:
                if em and em in {C.normalize_email(x.email) for x in g}:
                    target = (g, notes)
                    if any(x.email and x.email != o.email and C.normalize_email(x.email) == em for x in g):
                        notes.append(f"merged duplicate {o.email!r} (same address, different case) from {o.source} {o.ref}")
                    break
            if target is None:
                for g, notes in groups:
                    for x in g:
                        same, score = C.same_person(o.name, x.name)
                        if same and _compatible(o, x):
                            target = (g, notes)
                            notes.append(f"merged {o.name!r} <{o.email}> into {x.name!r} <{x.email}> "
                                         f"(name match {score:.0f}, compatible domain/company)")
                            break
                    if target:
                        break
            if target:
                target[0].append(o)
            else:
                groups.append(([o], []))
        return groups

    def build(self) -> tuple[list[Attendee], list[Issue]]:
        d = self.web.directory()
        firms = {f["domain"].lower(): f["firm"] for f in d.firms}
        firm_names = list(firms.values())
        by_email = {p["email"].lower(): p for p in d.people}
        known_domains = set(firms) | C.FREE_MAIL

        attendees, issues = [], []
        for obs, notes in self._group():
            cal = [o for o in obs if o.source == "calendar"]
            emails = list(dict.fromkeys(e for e in (C.normalize_email(o.email) for o in obs) if e))
            key = C.normalize_email(cal[0].email) if cal else (emails[0] if emails else f"name:{C.name_key(obs[0].name)}")
            # Accepted email corrections take part in the directory join too.
            emails += [C.normalize_email(c.new) for c in self.corrections
                       if c.person == key and c.accepted and c.field == "email"]
            a = Attendee(key=key, name=None, email=None, phone=None, company=None, title=None, rsvp="no response",
                         on_invite=bool(cal), directory_match=None, notes=list(notes),
                         observation_ids=[o.obs_id for o in obs])
            iss = _IssueSink(key, issues)

            # Reference join: by email first, else by name.
            ref = next((by_email[e] for e in emails if e in by_email), None)
            if ref:
                a.directory_match = "email"
            else:
                best = max(((p, C.same_person(p["name"], o.name)) for p in d.people for o in obs if o.name),
                           key=lambda t: t[1][1], default=None)
                if best and best[1][0] and best[1][1] >= 92:
                    ref, a.directory_match = best[0], "name"

            self._resolve_name(a, obs, ref)
            self._resolve_email(a, obs, emails, ref, cal)
            self._resolve_phone(a, obs, ref)
            self._resolve_company(a, obs, ref, firms, firm_names)
            self._resolve_title(a, obs, ref)
            self._resolve_rsvp(a, obs)
            self._apply_corrections(a)
            self._validate(a, obs, ref, firms, known_domains, iss)
            attendees.append(a)

        for i in issues:
            if i.issue_id in self.human_review:
                i.status, i.review_note = "human_review", self.human_review[i.issue_id]
        attendees.sort(key=lambda x: (x.name or x.key).lower())
        return attendees, issues

    # --- field resolution (source priority: directory > email > calendar) --------------

    def _resolve_name(self, a: Attendee, obs: list[Observation], ref: dict | None) -> None:
        stated = [o for o in obs if o.name and not o.name_inferred]
        if ref:
            a.name, a.sources["name"] = ref["name"], "directory"
        elif stated:
            pick = sorted(stated, key=lambda o: (o.source != "email", -len(o.name)))[0]
            a.name, a.sources["name"] = C.normalize_name(pick.name), f"{pick.source}:{pick.ref}"
        elif obs[0].name:
            a.name, a.sources["name"] = obs[0].name, "inferred from email address"
        variants = {C.normalize_name(o.name) for o in stated} - {a.name}
        if variants:
            a.notes.append(f"name variants normalised to {a.name!r}: {sorted(v for v in variants if v)}")

    def _resolve_email(self, a, obs, emails, ref, cal) -> None:
        if ref and ref["email"].lower() in emails:
            a.email, a.sources["email"] = ref["email"].lower(), "directory (confirmed)"
        elif cal:
            a.email, a.sources["email"] = C.normalize_email(cal[0].email), f"calendar:{cal[0].ref}"
        elif emails:
            src = next(o for o in obs if C.normalize_email(o.email) == emails[0])
            a.email, a.sources["email"] = emails[0], f"{src.source}:{src.ref}"
        a.alt_emails = [e for e in emails if e != a.email]

    def _resolve_phone(self, a, obs, ref) -> None:
        candidates = []
        for o in sorted(obs, key=lambda o: o.at or "", reverse=True):
            if o.phone:
                e164, problem = C.normalize_phone(o.phone)
                if problem:
                    a.notes.append(f"ignored {problem} from {o.source}:{o.ref}")
                else:
                    candidates.append((e164, f"{o.source}:{o.ref}"))
        if ref and ref.get("phone"):
            e164, _ = C.normalize_phone(ref["phone"])
            a.phone, a.sources["phone"] = e164, "directory"
            others = {c[0] for c in candidates} - {e164}
            if others:
                a.notes.append(f"phone {sorted(others)} from email differs from directory {e164}; kept directory value")
        elif candidates:
            a.phone, a.sources["phone"] = candidates[0]
            if len({c[0] for c in candidates}) > 1:
                a.notes.append(f"conflicting phones {sorted({c[0] for c in candidates})}; kept most recent")

    def _resolve_company(self, a, obs, ref, firms, firm_names) -> None:
        raw = [o.company for o in obs if o.company]
        if ref:
            a.company, a.sources["company"] = ref["firm"], "directory"
        elif raw:
            a.company, score = C.canonical_company(raw[0], firm_names)
            a.sources["company"] = f"email (canonicalised, match {score:.0f})" if a.company in firm_names else "email"
        elif a.email and C.email_domain(a.email) in firms:
            a.company, a.sources["company"] = firms[C.email_domain(a.email)], "directory (firm email domain)"
        variants = {r for r in raw if C.canonical_company(r, firm_names)[0] != r}
        if variants:
            a.notes.append(f"company variants {sorted(variants)} canonicalised to {a.company!r}")

    def _resolve_title(self, a, obs, ref) -> None:
        raw = [o.title for o in obs if o.title]
        if ref and ref.get("title"):
            a.title, a.sources["title"] = ref["title"], "directory"
            other = {t for t in raw if t.lower() != ref["title"].lower()}
            if other:
                a.notes.append(f"title {sorted(other)} in email replaced by directory title {ref['title']!r}")
        elif raw:
            a.title, a.sources["title"] = raw[0], "email"

    def _resolve_rsvp(self, a, obs) -> None:
        """Calendar response is the baseline; a later, explicit email statement wins."""
        cal = next((o for o in obs if o.source == "calendar"), None)
        status, src = (cal.attendance, f"calendar:{cal.ref}") if cal else ("no response", "none")
        stated = sorted((o for o in obs if o.source == "email" and o.attendance not in (None, "unknown")),
                        key=lambda o: o.at or "")
        for o in stated:
            new = o.attendance.replace("requested_by_colleague", "requested by colleague")
            if new == "requested by colleague" and status not in ("no response",):
                continue
            if new != status:
                if cal or status != "no response":
                    a.notes.append(f"RSVP changed {status!r} -> {new!r} by {o.source}:{o.ref} ({o.at[:10] if o.at else ''}): "
                                   f"\"{o.evidence[:80]}\"")
                status, src = new, f"{o.source}:{o.ref}"
        a.rsvp, a.sources["rsvp"] = status, src

    def _apply_corrections(self, a: Attendee) -> None:
        for c in self.corrections:
            if c.person == a.key and c.accepted:
                value = C.normalize_email(c.new) if c.field == "email" else (
                    C.normalize_phone(c.new)[0] or c.new if c.field == "phone" else c.new)
                setattr(a, c.field, value)
                a.sources[c.field] = f"correction {c.corr_id} ({c.evidence})"
                if c.field == "email" and c.old and c.old not in a.alt_emails:
                    a.alt_emails.append(c.old)

    # --- validation ------------------------------------------------------------------

    def _validate(self, a, obs, ref, firms, known_domains, iss) -> None:
        if not a.email:
            iss.add("email", "error", "missing_email", "no email address in any source")
        elif not C.email_is_valid(a.email):
            iss.add("email", "error", "invalid_email", f"'{a.email}' is not a valid email address")
        else:
            dom = C.email_domain(a.email)
            if dom not in known_domains:
                typo = C.domain_typo_of(dom, known_domains)
                dns = self.web.mx_lookup(dom)
                if typo:
                    sugg = ref["email"] if ref and C.email_domain(ref["email"]) == typo else a.email.replace(dom, typo)
                    iss.add("email", "error", "email_domain_typo",
                            f"domain '{dom}' looks like a typo of '{typo}' (DNS: {dns['status']})",
                            suggestion=f"{sugg}" + (" (directory)" if ref and sugg == ref['email'] else " (unconfirmed)"))
                elif dns["status"] != "ok":
                    iss.add("email", "error", "email_domain_undeliverable",
                            f"domain '{dom}' cannot receive mail (DNS: {dns['status']})")
        cal = next((o for o in obs if o.source == "calendar"), None)
        if cal and a.email and C.normalize_email(cal.email) != a.email:
            iss.add("invite", "warning", "invite_address_mismatch",
                    f"calendar invite went to {cal.email} but the verified address is {a.email}; "
                    f"the invite may not have reached them", suggestion=f"update invite to {a.email}")
        if not a.on_invite:
            iss.add("invite", "warning", "not_on_invite",
                    "registered by email but not on the calendar invite", suggestion="add to invite")
        if a.directory_match is None:
            iss.add("directory", "warning", "not_in_directory", "not found in the partner directory - identity unverified")
        elif a.directory_match == "name" and a.email and a.email != ref["email"].lower():
            iss.add("email", "error", "email_differs_from_directory",
                    f"matched directory entry by name, but directory email is {ref['email']}",
                    suggestion=f"{ref['email']} (directory)")
        if not a.phone:
            iss.add("phone", "warning", "missing_phone", "no valid phone number in any source")
        if not a.company:
            iss.add("company", "warning", "missing_company", "company unknown")
        elif a.company not in firms.values():
            iss.add("company", "info", "company_not_accredited", f"'{a.company}' is not an accredited partner firm")
        if "name" in a.sources and a.sources["name"] == "inferred from email address":
            iss.add("name", "warning", "name_inferred", f"name '{a.name}' inferred from the email address")
        _dedupe_same_field_errors(iss)

    # --- agent actions ----------------------------------------------------------------

    def propose_correction(self, person: str, field_name: str, value: str, evidence: str, rationale: str) -> Correction:
        attendees, _ = self.build()
        a = next((x for x in attendees if x.key == person), None)
        cid = f"C{len(self.corrections) + 1}"
        if a is None:
            c = Correction(cid, person, field_name, None, value, evidence, rationale, False, f"unknown person '{person}'")
        elif field_name not in FIELDS:
            c = Correction(cid, person, field_name, None, value, evidence, rationale, False, f"field must be one of {FIELDS}")
        else:
            ok, check = self._evidence_supports(field_name, value, evidence)
            c = Correction(cid, person, field_name, getattr(a, field_name), value, evidence, rationale, ok, check)
        self.corrections.append(c)
        return c

    def _evidence_supports(self, field_name: str, value: str, evidence: str) -> tuple[bool, str]:
        """A correction is accepted only if the new value literally appears in the
        cited evidence. DNS is deliberately NOT accepted for mailbox corrections:
        it proves a domain accepts mail, not that a person's mailbox exists."""
        def norm(s):
            return re.sub(r"\s+", " ", (s or "").lower()).strip()

        def phone_eq(x, y):
            return C.normalize_phone(x)[0] is not None and C.normalize_phone(x)[0] == C.normalize_phone(y)[0]

        if evidence == "directory":
            for p in self.web.directory().people:
                for v in p.values():
                    if norm(v) == norm(value) or (field_name == "phone" and phone_eq(v, value)):
                        return True, f"value found in directory entry for {p['name']}"
            return False, "value not found in the partner directory"
        if evidence.startswith("email:"):
            msg = self.messages.get(evidence.split(":", 1)[1])
            if not msg:
                return False, f"{evidence} has not been read in this run"
            text = f"{msg['from']} {msg['body']}"
            if norm(value) in norm(text):
                return True, f"value found in {evidence}"
            if field_name == "phone" and re.sub(r"\D", "", value)[-9:] in re.sub(r"\D", "", text):
                return True, f"phone digits found in {evidence}"
            return False, f"value not found in {evidence}"
        if evidence.startswith("dns:"):
            return False, ("DNS only proves the domain accepts mail, not that this mailbox exists - "
                           "escalate for human confirmation instead")
        return False, "evidence must be 'directory' or 'email:<message_id>'"

    def flag_for_review(self, issue_id: str, note: str) -> str:
        _, issues = self.build()
        if issue_id not in {i.issue_id for i in issues}:
            return f"unknown issue_id {issue_id}"
        self.human_review[issue_id] = note
        return "flagged for human review"

    # --- export ------------------------------------------------------------------------

    def table(self) -> tuple[list[dict], list[Issue]]:
        attendees, issues = self.build()
        rows = []
        for a in attendees:
            mine = [i for i in issues if i.person == a.key]
            rows.append({
                "name": a.name, "email": a.email, "phone": a.phone, "company": a.company, "title": a.title,
                "rsvp": a.rsvp, "on_invite": a.on_invite,
                "directory_verified": {"email": "yes (email)", "name": "partial (name)"}.get(a.directory_match, "no"),
                "status": _row_status(mine),
                "flags": "; ".join(f"[{i.severity}] {i.code}" + (" (human review)" if i.status == "human_review" else "")
                                   for i in mine),
                "alt_emails": ", ".join(a.alt_emails),
                "key": a.key,
            })
        return rows, issues


class _IssueSink:
    def __init__(self, person: str, issues: list[Issue]):
        self.person, self.issues, self.mine = person, issues, []

    def add(self, field_name, severity, code, message, suggestion=None):
        i = Issue(f"{self.person}:{code}", self.person, field_name, severity, code, message, suggestion)
        self.issues.append(i)
        self.mine.append(i)


def _dedupe_same_field_errors(iss: _IssueSink) -> None:
    """A typo'd domain also makes the name-matched directory email differ; report the root cause once."""
    codes = {i.code for i in iss.mine}
    if "email_domain_typo" in codes and "email_differs_from_directory" in codes:
        drop = next(i for i in iss.mine if i.code == "email_differs_from_directory")
        iss.mine.remove(drop)
        iss.issues.remove(drop)


def _compatible(a: Observation, b: Observation) -> bool:
    da, db = C.email_domain(C.normalize_email(a.email)), C.email_domain(C.normalize_email(b.email))
    if da and db:
        return da == db
    ca, cb = C.company_key(a.company), C.company_key(b.company)
    if ca and cb:
        return ca in cb or cb in ca
    return True


def _row_status(issues: list[Issue]) -> str:
    open_ = [i for i in issues if i.status == "open"]
    if any(i.severity == "error" for i in open_):
        return "needs fix"
    if any(i.status == "human_review" for i in issues):
        return "human review"
    if any(i.severity == "warning" for i in open_):
        return "flagged"
    return "verified"


def issue_dict(i: Issue) -> dict:
    return asdict(i)
