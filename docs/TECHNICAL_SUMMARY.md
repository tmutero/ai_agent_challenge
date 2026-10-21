# AI Agent Technical Challenge – Technical Summary

One Streamlit web app, two parts, shared Python core. Built in code rather than a no-code tool so that every
reasoning step is inspectable, testable and measured.

| | Part 1 – RAG over structured docs | Part 2 – Attendee reconciliation agent |
|---|---|---|
| LLM | OpenAI `gpt-5.6` (structured outputs) | OpenAI `gpt-5.6` (Responses API tool calling + reasoning summaries) |
| Retrieval / tools | Qdrant (dense `text-embedding-3-small` + BM25 sparse, RRF fusion), pandas | Google Calendar, Gmail, HTTP scrape of a public directory, DNS MX lookups |
| Deterministic core | record-level ingestion, plan validation, aggregation engine, fact verifier | normalisation, de-duplication, reference join, validation rules, evidence checker |
| UI | chat + "How this answer was produced" trace + KB browser + eval results | live agent trace + cleansed table + review queue + corrections log + provenance + CSV |
| Tests | 30 offline tests (no API key: fake embedder, in-memory Qdrant, fixture tools) | |

**Design principle used throughout: the LLM proposes, deterministic code verifies.** The model is used where language
understanding is needed (rewriting queries, reading messy emails, deciding what to investigate). Values, joins, rules
and acceptance of changes are handled by code that can be unit-tested and that produces the same output for the same
input.

---

## Part 1 – RAG deployment

### Knowledge base: uploaded documents
Users upload their own files in the **Documents** tab: CSV/TSV, Excel (one table per sheet), JSON, PDF, Word (.docx)
and TXT/Markdown, up to 20 MB each. Each browser session has its own document set and index. Re-indexing after an
upload only embeds new chunks, because embeddings are cached by content hash. A file that fails to parse is reported
with its error and does not block the others.

The loaders contain nothing dataset-specific. For each table they infer:
* the **key column**: the first column with unique, non-empty values, preferring text. It becomes the record ID.
* the **title column**: a mostly-unique, short text column such as a name. It becomes the row heading.
* the **filterable columns**: text columns with at most 30 distinct short values. They become payload filters, and
  their allowed values are listed to the query planner.
* **readable labels**: `ter_pct` becomes "TER %", and `min_lump_sum_zar` becomes "Minimum lump sum (ZAR)".

For JSON, the main list of records becomes a table, each nested list of objects becomes a child table that cites its
parent record, and top-level scalars are kept as context on every record. PDF and Word tables become tables, and the
remaining text becomes paragraphs.

Four **sample documents** for a fictional fund manager are preloaded and go through the same upload path. They can be
removed or re-added with one click.

#### Sample documents
| File | Format | Content | Records |
|---|---|---|---|
| `funds.csv` | CSV | fund factsheet: TER, minimums, risk rating, Reg 28, 1/3/5y returns (some deliberately *not reported*) | 14 |
| `fee_schedule.json` | nested JSON | platform products with tiered admin fees, switch rules, contribution limits | 4 products / 10 tiers |
| `fica_requirements.csv` | CSV | onboarding documents per client type | 6 |
| `investment_policy.pdf` | **tabular PDF** + prose | suitability matrix, asset-class limits table (spans a page break), breach rules | 13 rows + 2 paragraphs |

### Pipeline
```
files -> structure-aware loaders ---------------------------> pandas DataFrames (aggregation engine)
           | one Document per record, with _source/_record_ref/_chunk_id on both sides
           v
        AtomicChunker -> OpenAI embeddings + BM25 -> Qdrant (payload: category, record_ref, metadata.fields.*)

question + chat history
   -> Query planner (LLM, strict schema): standalone rewrite, intent, dataset routing, column filters, sub-queries, aggregation spec
   -> validate_plan (code): drop unknown datasets/columns/filter values (logged as "planner corrections")
   -> intent router
        lookup/catalog : one hybrid search per routed dataset (filters only constrain their own dataset)
        compare        : one search per sub-query x dataset, merged
        aggregate      : pandas over the FULL table (min/max/mean/sum/count/list) + the source rows as citations
        out_of_scope   : polite refusal, no retrieval
   -> evidence gate: best dense cosine >= 0.30, else retry unrouted, else refuse
   -> grounded generation (LLM, strict schema): markdown answer with [n] citations + atomic facts {value, passage}
   -> fact verifier (code): every fact's value must literally/numerically occur in the passage it cites
```

**Ingestion, chunking and embedding.** Structured data is chunked **per record**, not by character count. A row split
across chunks would separate values from their column headers, and a chunk spanning several rows blurs which fund a
number belongs to. Each row is rendered as readable `Label: value` lines; `ter_pct` becomes "Total expense ratio (TER)
%" so that it embeds and reads well. Missing cells become an explicit `not reported`. The PDF loader uses `pdfplumber`
table detection. It stitches a table that continues across a page break without its header row, and groups the
remaining prose into paragraphs using vertical gaps between lines. Six "dataset overview" chunks answer "what data do
you have?".

**Query transformation and prompt control.**
* *Context retention*: the planner rewrites follow-ups into standalone queries ("And its 5-year return?" →
  "What is the 5-year return of the Kestrel Bond Fund?"; "which of those…" → a filter on the three funds just discussed).
* *Intent routing*: aggregate questions ("cheapest", "average", "how many") go to pandas, because top-k vector search
  cannot rank 14 funds by TER unless they all happen to be retrieved.
* *Metadata pre-filtering*: the catalog lists the allowed values for each filterable column, and anything else the
  model proposes is dropped and shown in the trace.
* *Structured answers*: comparisons render as tables. Values are copied exactly. "Not reported" is stated, never
  estimated.

**Relevance, context retention and traceability.** Every answer shows its sources (file + row/table reference).
An expander shows the plan, the planner corrections, the routes and filters, every retrieved chunk with dense/BM25/RRF
scores, the deterministic aggregation table, the exact passages sent to the model, a per-fact verification table and
timings. A badge reports "N/N facts verified against cited source". A second check flags any number stated in the
answer prose that appears in no passage (nor in the question), which covers numbers the model did not declare as
facts.

### Evaluation (20 labelled questions, `python -m part1_rag.evaluate`)
| Configuration | Recall@5 | MRR | NDCG |
|---|---:|---:|---:|
| dense only | 0.85 | 0.80 | 0.81 |
| hybrid (dense + BM25, RRF) | 0.95 | 0.86 | 0.88 |
| hybrid + lexical rerank | 0.85 | 0.69 | 0.73 |
| **shipped: hybrid + query planner/routing** | **1.00** | **1.00** | **1.00** |

The evaluation drove the configuration choices. BM25 helps with codes and exact terms ("Capped SWIX", "TFSA"). The
pipeline's built-in lexical reranker *hurt* on short records, so it is disabled. Planning and routing contribute the
largest gain.

---

## Part 2 – Agent with tooling, extraction and verification

**Task:** "Prepare the verified attendee list for the Q4 Investor Roadshow in Cape Town." The agent reconciles the
calendar invite with RSVP/registration emails, verifies each person against a public partner directory and DNS, fixes
what evidence supports, and flags everything else.

### Tools (all behind interfaces, with a live and an offline implementation)
| Tool | Live | Offline |
|---|---|---|
| Calendar: `search_calendar_events`, `get_event_attendees` | Google Calendar API | scenario fixture |
| Email: `search_emails`, `read_email`, `extract_contacts`, draft creation (UI button, never sent) | Gmail API | fixture mailbox with a Gmail-syntax query evaluator |
| Web: `lookup_directory` (HTTP scrape + BeautifulSoup), `check_email_domain` (DNS MX) | real HTTP + real DNS | same |
| Workspace: `build_attendee_list`, `apply_correction`, `flag_for_human_review`, `finish` | deterministic engine | same |

Scenario dates are relative to today (the event is about 3 weeks out, the emails arrived in the past week), so the demo never goes stale. The live and offline modes are seeded from **one scenario file** (`google_setup seed` writes the same event and emails
into a Google account; invites use `sendUpdates=none`, emails are *inserted*, never sent, and firm domains use the
reserved `.example` TLD).

### Orchestration
```
LLM (Responses API, reasoning summaries on)           deterministic code
--------------------------------------------           ------------------------------------------------
plan -> find event -> load attendees  ---------------> exclude organiser / rooms / internal staff
search mail (several queries) -> extract_contacts ---> LLM extraction -> grounding check (value must occur in email)
build_attendee_list  --------------------------------> normalise (E.164, names, firms) -> entity resolution
                                                        -> directory join (email, else name) -> field priority
                                                        -> RSVP timeline -> validation rules -> issues
for each issue: investigate (directory / DNS / mail)
   apply_correction(value, evidence) ----------------> accepted only if value occurs in cited directory row/email;
                                                        DNS is rejected as evidence for a mailbox
   or flag_for_human_review(note)
rebuild -> confirm no new issues -> finish(summary)
```

### Data handling quality
* **Normalisation**: E.164 phones (handles `+27 (0)83…` and `(021) 555-0198`, and rejects truncated SA numbers that the
  `phonenumbers` library alone accepts), name casing with particles and apostrophes, firm canonicalisation by fuzzy match
  to the directory ("Fynbos Wealth" → "Fynbos Wealth Partners").
* **De-duplication**: by normalised email ("LERATO.MOKOENA@…" from the registration table), then by name plus a
  compatible domain or company ("J.Botha@KarooCapital…" with no display name ↔ "Johan Botha <johan.botha@…>"). Every
  merge is logged.
* **Conflict resolution**: source priority directory > email > calendar for identity fields. For RSVP, the latest
  explicit statement wins (Ayanda: tentative in the calendar → *declined* by email two days later).
* **Validation**: domain typo detection (`highveldfo.exmaple`, `gmial.com`) with a DNS confirmation, undeliverable
  domains, missing phone or company, not on the invite (Kabelo registered by email only), invite sent to a non-verified
  alias, and identity not in the directory.
* **Error handling**: tool errors are returned to the model instead of crashing the run. There is a step budget. The
  directory fetch falls back to a bundled copy and records that it did. Values the extraction invents are rejected and
  listed.

### Transparency: how the agent verifies or corrects itself
* Each reasoning summary, tool call, argument and result is streamed live and kept in a "Full trace" tab.
* Corrections are **claims with evidence** that code checks. In testing, the agent fixed Priya's typo'd domain from the
  directory, which was accepted and re-validated. It did *not* rewrite Grace's `gmial.com` address to `gmail.com`, because
  no source confirms that mailbox; the rules engine would also reject DNS as evidence for it.
* After a correction the row is re-validated by the same rules. Fixing Priya's email surfaced a *new* warning, "the
  invite still points to the typo address", which the agent then escalated.
* Each field in the output carries its provenance (`directory`, `email:msg-003`, `correction C1 (directory)` …).

**Output**: a cleansed table (status ✅ verified / ⚠️ flagged / 👤 human review / ❌ needs fix), CSV export, an issues
and review queue, a corrections log, a per-attendee provenance view and the excluded records. A Gmail draft can be
created with one click; it is never sent automatically.

---

## Limitations and what I'd do with more time
* **Scale (Part 1)**: aggregation is limited to one table at a time with equality filters. Next steps would be a
  constrained text-to-SQL path over DuckDB for joins and range filters (the existing SQL-explainer work fits here), and
  incremental ingestion with a persistent Qdrant server instead of in-memory.
* **Reranking**: replace the lexical reranker with a cross-encoder or Voyage rerank and re-run the evaluation. Grow the
  evaluation set to include end-to-end answer correctness (LLM-as-judge plus exact-value checks).
* **Agent robustness**: run evaluations of the agent itself (expected final table vs. scenario ground truth across N
  runs to measure variance), cache extractions, and parallelise extraction calls (about 100k tokens per run today).
* **Write-back**: with approval, update the calendar invite (add Kabelo, fix Priya's and Johan's addresses) through a
  human-in-the-loop confirmation step.
* **Security and ops**: OAuth tokens in a secrets manager, per-user auth on the hosted app, PII redaction in the
  traces, and structured logs / OpenTelemetry spans per tool call.
* **Hosted Google mode**: Google OAuth in "Testing" mode issues refresh tokens that expire after 7 days. Live
  Gmail/Calendar is therefore demoed locally and the hosted app defaults to the offline scenario. Production would use
  a verified OAuth app or a Workspace service account with domain-wide delegation.
