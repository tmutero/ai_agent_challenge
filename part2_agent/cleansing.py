"""
Deterministic normalisation and validation rules -- pure functions, no I/O, no LLM.

The agent decides *what to look at*; these rules decide *what is valid*. Keeping
them deterministic means the same input always produces the same cleansed output,
and every rule is unit-tested.
"""

from __future__ import annotations

import re
import unicodedata

import phonenumbers
from rapidfuzz import fuzz, process

FREE_MAIL = {"gmail.com", "outlook.com", "hotmail.com", "yahoo.com", "icloud.com", "live.com"}
_EMAIL = re.compile(r"^[a-z0-9._%+'-]+@[a-z0-9.-]+\.[a-z]{2,}$")
_PARTICLES = {"van", "der", "de", "du", "le", "la", "von", "da", "dos"}
_SUFFIXES = re.compile(r"\b(pty|ltd|limited|inc|llc|\(pty\)|cc)\b\.?", re.I)


# --- email -----------------------------------------------------------------------

def normalize_email(raw: str | None) -> str | None:
    if not raw:
        return None
    m = re.search(r"<([^>]+)>", raw)            # "Name <addr>" -> addr
    value = (m.group(1) if m else raw).strip().strip(".,;").lower()
    return value or None


def email_is_valid(email: str | None) -> bool:
    return bool(email and _EMAIL.match(email))


def email_domain(email: str | None) -> str | None:
    return email.split("@", 1)[1] if email and "@" in email else None


def domain_typo_of(domain: str, known: set[str], threshold: float = 85) -> str | None:
    """Closest known domain if ``domain`` looks like a misspelling of it."""
    if not domain or domain in known or not known:
        return None
    match = process.extractOne(domain, known, scorer=fuzz.ratio)
    return match[0] if match and match[1] >= threshold else None


# --- phone -------------------------------------------------------------------------

def normalize_phone(raw: str | None, region: str = "ZA") -> tuple[str | None, str | None]:
    """Return (E.164, problem). Handles '+27 (0)83 ...' trunk prefixes and local
    formats. South African numbers must have a 9-digit national number -- the
    phonenumbers metadata alone accepts some truncated numbers."""
    if not raw or not raw.strip():
        return None, "missing"
    cleaned = raw.replace("(0)", "")
    try:
        num = phonenumbers.parse(cleaned, region)
    except phonenumbers.NumberParseException:
        return None, f"unparseable phone '{raw}'"
    if not phonenumbers.is_valid_number(num):
        return None, f"invalid phone '{raw}'"
    if num.country_code == 27 and len(str(num.national_number)) != 9:
        return None, f"invalid phone '{raw}' (SA numbers need 9 digits after the country code)"
    return phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164), None


# --- names -------------------------------------------------------------------------

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize_name(raw: str | None) -> str | None:
    """'LERATO MOKOENA' -> 'Lerato Mokoena', 'michael oconnor' -> 'Michael Oconnor',
    "sarah VAN DER merwe" -> 'Sarah van der Merwe'. Apostrophes are preserved."""
    if not raw or not raw.strip():
        return None
    words = re.sub(r"\s+", " ", raw.strip()).split(" ")
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i > 0 and lw in _PARTICLES:
            out.append(lw)
        else:
            out.append("'".join(p[:1].upper() + p[1:] for p in lw.split("'")))
    return " ".join(out)


def name_key(name: str | None) -> str:
    """Comparison key: lowercase, no accents/punctuation ("O'Connor" == "oconnor")."""
    return re.sub(r"[^a-z ]", "", _strip_accents(name or "").lower()).strip()


def name_from_email(email: str) -> str | None:
    """'j.botha@x' -> 'J Botha' (used only when no display name exists; flagged)."""
    local = email.split("@", 1)[0]
    parts = [p for p in re.split(r"[._-]+", local) if p.isalpha()]
    return normalize_name(" ".join(parts)) if len(parts) >= 2 else None


def same_person(name_a: str | None, name_b: str | None) -> tuple[bool, float]:
    """Exact-ish match on full names, or initial+surname compatibility ('J Botha' ~ 'Johan Botha')."""
    a, b = name_key(name_a).split(), name_key(name_b).split()
    if not a or not b:
        return False, 0.0
    score = fuzz.token_sort_ratio(" ".join(a), " ".join(b))
    if score >= 92:
        return True, score
    if a[-1] == b[-1] and a[0][0] == b[0][0] and (len(a[0]) == 1 or len(b[0]) == 1):
        return True, 90.0
    return False, score


# --- companies -------------------------------------------------------------------

def company_key(name: str | None) -> str:
    return re.sub(r"\s+", " ", _SUFFIXES.sub("", (name or "").lower())).strip(" .,")


def canonical_company(raw: str | None, firms: list[str], threshold: float = 85) -> tuple[str | None, float]:
    """Map a free-text company to a directory firm ('Fynbos Wealth' -> 'Fynbos Wealth Partners')."""
    if not raw:
        return None, 0.0
    if not firms:
        return raw.strip(), 0.0
    choices = {company_key(f): f for f in firms}
    match = process.extractOne(company_key(raw), list(choices), scorer=fuzz.token_set_ratio)
    if match and match[1] >= threshold:
        return choices[match[0]], match[1]
    return raw.strip(), match[1] if match else 0.0


# --- RSVP ------------------------------------------------------------------------

RSVP_FROM_CALENDAR = {"accepted": "attending", "declined": "declined", "tentative": "tentative",
                      "needsAction": "no response"}
