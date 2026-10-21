# AI Agent Technical Challenge – Technical Summary

**Live app:** <https://aiagentchallenge.streamlit.app> (login details supplied separately)

One Streamlit web app with two parts on a shared Python core. I built it in code rather than a no-code tool so that
every reasoning step can be inspected, tested and measured.

| | Part 1 – RAG over structured documents | Part 2 – Attendee reconciliation agent |
|---|---|---|
| **LLM** | OpenAI `gpt-5.6`, structured outputs | OpenAI `gpt-5.6`, Responses API tool calling with reasoning summaries |
| **Retrieval / tools** | Qdrant hybrid search (dense `text-embedding-3-small` + BM25, RRF fusion), pandas | Google Calendar, Gmail, HTTP scrape of a public directory, DNS MX lookups |
| **Deterministic core** | Record-level ingestion, plan validation, aggregation engine, fact verifier | Normalisation, de-duplication, reference join, validation rules, evidence checker |
| **UI** | Chat, per-answer trace, document manager, evaluation results | Live agent trace, cleansed table, review queue, corrections log, provenance, CSV export |

## Design principle: the LLM proposes, deterministic code verifies

The model handles the parts that need language understanding: rewriting queries, reading messy emails and deciding
what to investigate. Code handles values, joins, rules and whether a change is accepted, so those steps are
unit-tested and give the same output for the same input. Both parts make this split visible in the UI.

---

## Part 1 – RAG over structured documents

Users upload CSV/TSV, Excel, JSON, PDF, Word or text files and ask questions in chat. Sample documents are preloaded
through the same upload path. The loaders are generic: they infer each table's key, title and filterable columns, so
no dataset-specific code is needed.

### Pipeline

```text
files -> structure-aware loaders -> one chunk per record ------> pandas tables (aggregation engine)
                                         |
                                         v
                       OpenAI embeddings + BM25 -> Qdrant (with filterable metadata)

question + chat history
   -> query planner (LLM, strict schema): standalone rewrite, intent, dataset routing, filters, aggregation spec
   -> plan validation (code): unknown datasets / columns / filter values are dropped and logged
   -> intent router
        lookup / compare : hybrid search per routed dataset or sub-query
        aggregate        : pandas over the full table, with the source rows as citations
        out of scope     : polite refusal, no retrieval
   -> evidence gate: refuse if the best match is below the relevance threshold
   -> grounded answer (LLM, strict schema): markdown with [n] citations + a list of atomic facts
   -> fact verifier (code): each fact must literally or numerically appear in the passage it cites
```

### Key design decisions

* **One chunk per record, not by character count.** Splitting a row separates values from their column headers, and
  merging rows blurs which record a number belongs to. Each record is rendered as readable `Label: value` lines, and
  missing values are stated explicitly rather than left blank.
* **Hybrid retrieval.** Dense embeddings capture meaning, and BM25 catches exact terms and codes that embeddings miss.
* **Aggregates are calculated, not retrieved.** Questions such as "cheapest", "average" or "how many" go to pandas over
  the full table. Top-k vector search cannot rank records it never retrieved.
* **Controlled planning.** The planner only sees allowed filter values. Anything it invents is removed by code and
  shown in the trace as a "planner correction".
* **Context retention.** Follow-up questions ("and its 5-year return?") are rewritten into standalone queries before
  retrieval.

### Traceability

Every answer lists its sources (file and row or table reference). A "How this answer was produced" panel shows the
plan, planner corrections, routes and filters, every retrieved chunk with its dense, BM25 and fused scores, any
aggregation table, the exact passages sent to the model, per-fact verification and timings. A badge reports how many
facts were verified against their cited source. A second check flags any number in the answer that appears in no
source passage.

### Evaluation (20 labelled questions, `python -m part1_rag.evaluate`)

| Configuration | Recall@5 | MRR | NDCG |
| --- | ---: | ---: | ---: |
| Dense only | 0.85 | 0.80 | 0.81 |
| Hybrid (dense + BM25, RRF) | 0.95 | 0.86 | 0.88 |
| Hybrid + lexical rerank | 0.85 | 0.69 | 0.73 |
| **Shipped: hybrid + query planner and routing** | **1.00** | **1.00** | **1.00** |

The evaluation drove the configuration. Hybrid search beat dense-only, and the lexical reranker *hurt* on short
records, so it is turned off. Planning and routing gave the largest gain.

---

## Part 2 – Attendee reconciliation agent

**Task:** prepare a verified attendee list for an upcoming investor event. The agent reconciles the calendar invite
with RSVP and registration emails, verifies each person against a public partner directory and DNS, corrects what
the evidence supports and flags everything else for a person to review.

### Tools

Every tool sits behind an interface with a live and an offline implementation, so the same agent runs against Google
or a built-in scenario.

| Tool | Live | Offline |
|---|---|---|
| Calendar: search events, get attendees | Google Calendar API | Scenario fixture |
| Email: search, read, extract contacts, create draft | Gmail API | Fixture mailbox with Gmail-style search |
| Web: directory lookup (HTTP + BeautifulSoup), email-domain check (DNS MX) | Real HTTP and DNS | Same |
| Workspace: build list, apply correction, flag for review, finish | Deterministic engine | Same |

### Orchestration

```text
LLM (plans, investigates, decides)                deterministic code (checks and builds)
----------------------------------                -----------------------------------------------
find event -> load attendees  ------------------> exclude organiser, rooms, internal staff
search mail -> extract contacts  ---------------> grounding check: extracted values must occur in the email
build attendee list  ---------------------------> normalise -> de-duplicate -> join to directory
                                                  -> resolve conflicts -> validation rules -> issues
for each issue: investigate (directory / DNS / mail)
   apply_correction(value, evidence)  ----------> accepted only if the value appears in the cited source
   or flag_for_human_review(note)
rebuild -> confirm no new issues -> finish
```

### Data quality

* **Normalisation:** phone numbers to international (E.164) format, consistent name casing, and company names matched
  to their canonical directory form.
* **De-duplication:** by normalised email, then by name with a compatible domain or company. Every merge is logged.
* **Conflict resolution:** identity fields follow a fixed source priority (directory > email > calendar). For RSVPs,
  the latest explicit statement wins.
* **Validation:** domain typos (confirmed with DNS), undeliverable domains, missing details, people who registered but
  aren't on the invite, and identities not found in the directory.
* **Error handling:** tool errors go back to the model instead of crashing the run, runs have a step budget, the
  directory fetch falls back to a bundled copy (and records that it did), and invented extraction values are rejected.

### How the agent verifies and corrects itself

* **Corrections are claims with evidence.** The agent must cite the directory row or email that supports a change,
  and code accepts it only if the new value appears in that source. DNS alone is never accepted as proof that a
  mailbox exists, so the agent can't "fix" an address by guessing.
* **Re-validation.** A corrected row goes through the same rules again, which can surface new issues for the agent to
  handle or escalate.
* **Provenance.** Every output field records where it came from (directory, a specific email, or a numbered
  correction).
* **Transparency.** Every reasoning summary, tool call, argument and result streams live and stays in a full-trace
  tab.

**Output:** a cleansed table (✅ verified / ⚠️ flagged / 👤 human review / ❌ needs fix), CSV export, an issues and
review queue, a corrections log, per-attendee provenance, the excluded records, and a one-click Gmail draft that is
never sent automatically.

---

## Engineering and deployment

* **Tests:** 30 offline tests that need no API key (fake embedder, in-memory Qdrant, fixture tools).
* **Hosting:** Streamlit Community Cloud, deployed from a private GitHub repository. Secrets (OpenAI key, app login,
  Google token) live in Streamlit secrets, never in the repo.
* **Access:** the app is behind a username and password login.
* **Live Google mode:** the hosted app uses a stored OAuth token and seeds the demo event and emails into the
  connected account automatically before the first live run, so no manual setup is needed.
* **Safe by design:** demo calendar invites are created without notifications, demo emails are inserted into the
  mailbox rather than sent, external companies use reserved `.example` domains, and drafts are never sent
  automatically.

---

## Limitations and next steps

* **Scale (Part 1):** aggregation works on one table at a time with equality filters. Next would be a constrained
  text-to-SQL path over DuckDB for joins and ranges, plus a persistent Qdrant server. The current index is in memory,
  so uploads don't survive an app restart.
* **Reranking and evaluation:** try a cross-encoder reranker, and extend the evaluation to end-to-end answer
  correctness (exact-value checks plus LLM-as-judge).
* **Agent robustness:** evaluate the agent itself (compare its final table with the scenario's expected result over
  many runs to measure variance), cache extractions and run them in parallel. A run uses about 100k tokens today.
* **Write-back:** with a person's approval, update the calendar invite with confirmed corrections.
* **Security and operations:** per-user sign-in, a dedicated secrets manager, PII redaction in traces, and
  structured logs or OpenTelemetry spans per tool call.
* **Google OAuth:** an unpublished ("Testing") OAuth app issues tokens that expire after 7 days. Production would use a
  verified OAuth app or a Workspace service account.
