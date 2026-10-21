"""
The autonomous attendee-reconciliation agent (OpenAI tool calling).

Division of labour:
  LLM     plans, chooses which sources to read, extracts people from messy emails,
          investigates open issues, proposes evidence-backed corrections, and
          decides what needs a human.
  Code    every tool, every normalisation/dedup/validation rule (workspace.build),
          and the evidence check on every correction.

Loop: model -> tool calls -> results -> model ... until it calls ``finish`` or the
step budget runs out. Every reasoning message, call and result is recorded as a
``Step`` so the UI can replay exactly what the agent did.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

from rapidfuzz import fuzz

from common.llm import chat_model, client

from .connectors import Toolset
from .extraction import extract_people
from .workspace import Workspace, issue_dict

SYSTEM = """You are an operations agent for Kestrel Asset Management's events team. Your job: produce a verified,
de-duplicated attendee list for one calendar event, with every remaining problem clearly flagged.

Work like this:
1. Find the event in the calendar and load its attendees.
2. Search the mailbox for emails about the event (RSVPs, apologies, registration lists). Search more than once
   with different keywords if useful. Run extract_contacts on every relevant email; skip unrelated ones.
3. Call build_attendee_list. It normalises, de-duplicates and validates everything against the public partner
   directory (web) and DNS, and returns open issues.
4. Investigate each open error/warning. Use lookup_directory, check_email_domain, search_emails/read_email to look for
   evidence. If a source contains the correct value, call apply_correction citing that source ("directory" or
   "email:<message_id>"). If no source can confirm a fix (e.g. a personal mailbox typo, a missing phone that appears
   nowhere), call flag_for_human_review with a one-line note - do NOT guess.
5. Rebuild after corrections to confirm they took effect and introduced no new issues.
6. Call finish with a short summary: counts, what you corrected (with evidence), and what needs a human.

Rules: never invent values; never modify the calendar or send email; keep each reasoning message to one or two
sentences stating what you are about to do and why. Today's date is {today}."""

TOOLS = [
    ("search_calendar_events", "Search the user's calendar for events in the next N days.",
     {"query": {"type": "string"}, "days_ahead": {"type": "integer"}}, ["query", "days_ahead"]),
    ("get_event_attendees", "Load an event's attendee list into the workspace (organiser and rooms are excluded).",
     {"event_id": {"type": "string"}}, ["event_id"]),
    ("search_emails", "Search the mailbox (Gmail query syntax). Returns id, from, subject, date, snippet.",
     {"query": {"type": "string"}}, ["query"]),
    ("read_email", "Read one email's full text.", {"message_id": {"type": "string"}}, ["message_id"]),
    ("extract_contacts", "Extract people + attendance from an email into the workspace (LLM extraction, "
     "then values not literally present in the email are rejected).", {"message_id": {"type": "string"}}, ["message_id"]),
    ("lookup_directory", "Fuzzy search of the public partner directory web page by name, email, firm or domain.",
     {"query": {"type": "string"}}, ["query"]),
    ("check_email_domain", "DNS MX lookup: can this domain receive email?", {"domain": {"type": "string"}}, ["domain"]),
    ("build_attendee_list", "Consolidate, normalise, de-duplicate and validate. Returns the table and open issues.",
     {}, []),
    ("apply_correction", "Correct one field of one attendee. Accepted only if new_value appears in the cited evidence.",
     {"person_key": {"type": "string"}, "field": {"type": "string", "enum": ["name", "email", "phone", "company", "title"]},
      "new_value": {"type": "string"}, "evidence": {"type": "string", "description": "'directory' or 'email:<message_id>'"},
      "rationale": {"type": "string"}}, ["person_key", "field", "new_value", "evidence", "rationale"]),
    ("flag_for_human_review", "Mark an issue as needing a human decision.",
     {"issue_id": {"type": "string"}, "note": {"type": "string"}}, ["issue_id", "note"]),
    ("finish", "End the run with a summary for the events team.", {"summary": {"type": "string"}}, ["summary"]),
]


def _tool_specs() -> list[dict]:
    return [{"type": "function", "name": n, "description": d, "strict": True,
             "parameters": {"type": "object", "properties": p, "required": r, "additionalProperties": False}}
            for n, d, p, r in TOOLS]


@dataclass
class Step:
    kind: str            # reasoning | call | result | error | final
    name: str = ""
    payload: object = None
    ms: int = 0


@dataclass
class AgentRun:
    goal: str
    steps: list[Step] = field(default_factory=list)
    summary: str = ""
    finished: bool = False
    workspace: Workspace | None = None
    event_summary: str = ""
    tokens: int = 0


class AttendeeAgent:
    def __init__(self, tools: Toolset, *, max_turns: int = 30, on_step: Callable[[Step], None] | None = None):
        self.tools = tools
        self.ws = Workspace(tools.web)
        self.max_turns = max_turns
        self.on_step = on_step or (lambda s: None)
        self.run_state: AgentRun | None = None

    # --- tool implementations ----------------------------------------------------------

    def search_calendar_events(self, query: str, days_ahead: int) -> object:
        start = date.today()
        return self.tools.calendar.search_events(query, f"{start}T00:00:00Z",
                                                 f"{start + timedelta(days=max(1, days_ahead))}T23:59:59Z")

    def get_event_attendees(self, event_id: str) -> object:
        event = self.tools.calendar.get_event(event_id)
        self.run_state.event_summary = f"{event['summary']} ({event['start'][:10]})"
        return {"event": {k: event[k] for k in ("id", "summary", "start", "location")}, **self.ws.add_calendar_event(event)}

    def search_emails(self, query: str) -> object:
        return self.tools.mail.search(query)

    def read_email(self, message_id: str) -> object:
        msg = self.tools.mail.get(message_id)
        self.ws.messages[message_id] = msg
        return msg

    def extract_contacts(self, message_id: str) -> object:
        if not self.ws.event:
            return {"error": "load the event first (get_event_attendees)"}
        msg = self.tools.mail.get(message_id)
        result = extract_people(msg, self.ws.event["summary"])
        ids = self.ws.add_extraction(msg, result)
        return {"relevant_to_event": result.relevant, "people": result.people, "observation_ids": ids,
                "rejected_by_grounding_check": result.rejected}

    def lookup_directory(self, query: str) -> object:
        d = self.tools.web.directory()
        q = query.lower()
        scored = sorted(((max(fuzz.WRatio(q, p[k].lower()) for k in ("name", "email", "firm")), p) for p in d.people),
                        key=lambda t: -t[0])
        firms = [f for f in d.firms if fuzz.WRatio(q, f["firm"].lower()) >= 80 or fuzz.ratio(q, f["domain"]) >= 80]
        return {"source": d.url, "fetched_via": d.fetched_via, "notes": d.notes,
                "people": [dict(p, match=round(s)) for s, p in scored[:3] if s >= 60], "firms": firms}

    def check_email_domain(self, domain: str) -> object:
        return self.tools.web.mx_lookup(domain.split("@")[-1])

    def build_attendee_list(self) -> object:
        rows, issues = self.ws.table()
        compact = [{k: r[k] for k in ("key", "name", "email", "phone", "company", "rsvp", "status")} for r in rows]
        open_issues = [{k: v for k, v in issue_dict(i).items() if k in ("issue_id", "severity", "message", "suggestion")}
                       for i in issues if i.status == "open" and i.severity != "info"]
        return {"attendees": compact, "open_issues": open_issues,
                "human_review": [i.issue_id for i in issues if i.status == "human_review"],
                "excluded": self.ws.excluded + self.ws.internal_excluded}

    def apply_correction(self, person_key, field, new_value, evidence, rationale) -> object:
        c = self.ws.propose_correction(person_key, field, new_value, evidence, rationale)
        return {"correction_id": c.corr_id, "accepted": c.accepted, "check": c.check, "old": c.old, "new": c.new}

    def flag_for_human_review(self, issue_id: str, note: str) -> object:
        return self.ws.flag_for_review(issue_id, note)

    # --- loop -----------------------------------------------------------------------------

    def _emit(self, step: Step) -> None:
        self.run_state.steps.append(step)
        self.on_step(step)

    def run(self, goal: str) -> AgentRun:
        """Responses API loop. ``previous_response_id`` carries the conversation
        (including the model's reasoning items) server-side between turns."""
        self.run_state = AgentRun(goal=goal, workspace=self.ws)
        instructions = SYSTEM.format(today=date.today().isoformat())
        pending: list[dict] = [{"role": "user", "content": goal}]
        previous = None
        for _ in range(self.max_turns):
            resp = client().responses.create(
                model=chat_model(), instructions=instructions, input=pending, tools=_tool_specs(),
                previous_response_id=previous, reasoning={"effort": "medium", "summary": "auto"})
            previous, pending = resp.id, []
            self.run_state.tokens += resp.usage.total_tokens if resp.usage else 0
            calls = []
            for item in resp.output:
                if item.type == "reasoning":
                    text = " ".join(s.text for s in (item.summary or []))
                    if text:
                        self._emit(Step("reasoning", payload=text))
                elif item.type == "message":
                    text = "".join(c.text for c in item.content if getattr(c, "text", None))
                    if text:
                        self._emit(Step("reasoning", payload=text))
                        self.run_state.summary = text
                elif item.type == "function_call":
                    calls.append(item)
            if not calls:
                break
            for call in calls:
                name, args = call.name, json.loads(call.arguments or "{}")
                self._emit(Step("call", name, args))
                if name == "finish":
                    self.run_state.summary, self.run_state.finished = args.get("summary", ""), True
                    continue
                t0 = time.perf_counter()
                try:
                    result = getattr(self, name)(**args)
                    self._emit(Step("result", name, result, int((time.perf_counter() - t0) * 1000)))
                except Exception as exc:  # noqa: BLE001 -- surface tool errors to the model, don't crash
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                    self._emit(Step("error", name, result))
                pending.append({"type": "function_call_output", "call_id": call.call_id,
                                "output": json.dumps(result, default=str)[:12000]})
            if self.run_state.finished:
                break
        else:
            self._emit(Step("error", "budget", {"error": f"stopped after {self.max_turns} turns"}))
        self._emit(Step("final", payload=self.run_state.summary))
        return self.run_state


DEFAULT_GOAL = ("Prepare the verified attendee list for the upcoming Q4 Investor Roadshow in Cape Town: reconcile the "
                "calendar invite with RSVP and registration emails, verify everyone against the public partner directory, "
                "fix what the evidence supports and flag anything missing or doubtful.")
