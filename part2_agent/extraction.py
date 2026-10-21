"""
LLM extraction of people from an email (signatures, prose, pasted tables) into a
strict schema -- followed by a deterministic grounding check that rejects any
extracted email/phone/name that does not literally occur in the message. The LLM
is good at reading messy text; it is not trusted to be the source of a value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from common.llm import structured


class ExtractedPerson(BaseModel):
    full_name: str | None
    email: str | None
    phone: str | None = Field(description="exactly as written, do not reformat")
    company: str | None
    job_title: str | None
    attendance: Literal["attending", "declined", "tentative", "requested_by_colleague", "unknown"] = Field(
        description="this person's stated intent for the event, if the email states it")
    evidence: str = Field(description="short verbatim quote from the email supporting this record")


class EmailExtraction(BaseModel):
    relevant_to_event: bool = Field(description="false if the email is not about the named event")
    people: list[ExtractedPerson]


SYSTEM = """You extract attendee/contact records from one email about a corporate event.

Rules:
- Extract every person the email gives details for: the sender (from the signature/From header) and anyone listed
  or mentioned as a potential attendee (e.g. rows of a pasted registration table, a colleague who wants to join).
- Do NOT extract the event organiser/recipient, generic mailboxes (events@, marketing@), or organiser staff who
  merely forward or send a list (they are not attendees).
- Copy values EXACTLY as written (keep original case, spacing and phone formatting). Use null when absent - never guess.
- attendance: what the email says about THIS person attending the named event. A colleague the sender asks to add is
  "requested_by_colleague". A registration-table row with no other statement is "attending".
- If the email is unrelated to the named event, set relevant_to_event=false and return no people.
"""


@dataclass
class ExtractionResult:
    message_id: str
    relevant: bool
    people: list[dict]
    rejected: list[str] = field(default_factory=list)   # values dropped by the grounding check


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def ground(person: ExtractedPerson, text: str) -> tuple[dict, list[str]]:
    """Keep only values that occur in the source text."""
    rec = person.model_dump()
    rejected = []
    low = text.lower()
    for key in ("full_name", "email", "company", "job_title"):
        v = rec.get(key)
        if v and re.sub(r"\s+", " ", v.lower()) not in re.sub(r"\s+", " ", low):
            rejected.append(f"{key}='{v}' not found in email text")
            rec[key] = None
    if rec.get("phone") and _digits(rec["phone"]) not in _digits(text):
        rejected.append(f"phone='{rec['phone']}' not found in email text")
        rec["phone"] = None
    return rec, rejected


def extract_people(message: dict, event_summary: str) -> ExtractionResult:
    text = f"From: {message['from']}\nDate: {message['date']}\nSubject: {message['subject']}\n\n{message['body']}"
    out = structured(SYSTEM, f"Event: {event_summary}\n\nEmail:\n{text}", EmailExtraction)
    people, rejected = [], []
    for p in out.people:
        rec, rej = ground(p, text)
        rejected += [f"{p.full_name or p.email}: {r}" for r in rej]
        if rec.get("full_name") or rec.get("email"):
            people.append(rec)
    return ExtractionResult(message["id"], out.relevant_to_event, people, rejected)
