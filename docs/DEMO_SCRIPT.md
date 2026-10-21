# Demo script (about 25 minutes)

## 0. Framing (2 min)
* Show the technical summary page: two parts, one app, and one principle: **the LLM proposes, deterministic code
  verifies**.
* Show the architecture tables, and say that the design is measured, tested (30 offline tests) and traceable.

## 1. Part 1 – RAG (10 min)
1. **Documents tab.** The four sample files are preloaded. Show the per-file parse report and the extracted tables:
   each row is one chunk *and* one DataFrame row with the same `_chunk_id`. Point out the PDF table that crosses a
   page break, and the JSON tier lists that became a child table.
   **Upload a file of your own live**, e.g. a CSV or Excel sheet the panel suggests. Show that the key column and the
   filterable columns were inferred, then ask a question about it.
2. Ask **"What is the TER of the Kestrel Bond Fund?"** and open *How this answer was produced*:
   * the plan: intent `lookup`, routed to `funds`
   * the retrieved chunks with dense, BM25 and RRF scores, and the evidence gate
   * the verification table: 0.55 was found in passage [1], and the badge also checks every number in the prose
3. Ask **"And its 5-year return?"** to show context retention: the standalone rewrite resolves "its".
4. Ask **"Which 3 SA Equity funds have the lowest TER?"**, which is routed to the aggregation engine. Explain why
   top-k search can't rank all the rows, then show the computed table.
5. Ask **"And which of those has the best 3-year return?"**. The planner turns "those" into a `fund_name` filter on
   the three funds.
6. Ask **"Can a conservative client invest in the property fund?"**. This is a cross-dataset answer: the fund risk
   rating, the suitability matrix row and the deviation-letter clause from the PDF prose.
7. Ask **"What's the 5-year return of the sustainable equity fund?"**. The answer is "not reported", with no guess.
8. Ask **"What will the JSE do next year?"**, which is refused as out of scope.
9. **Evaluation tab.** Hybrid beat dense-only, the reranker hurt so it is off, and planning brings recall@5 to 1.0.

## 2. Part 2 – Agent (10 min)
1. Show the inputs: the Google Calendar event with 10 attendees (organiser and room included), the Gmail inbox (an
   RSVP, a Johan email with a truncated phone number, an apology, a forwarded registration table, a newsletter) and
   the public **partner directory page** at `/app/static/partner_directory.html`.
2. Click **Run agent** with the data source set to Google (live), or Fixture as the fallback. Narrate the live trace:
   reasoning, then calls, then results.
3. When it calls `build_attendee_list`, show the issues the rules found.
4. Show the self-correction. The LLM's path varies between runs (search order, which issues it investigates first), so narrate what it does rather than a fixed sequence. Things to look for:
   * **Priya**: typo'd domain → directory lookup → `apply_correction` with evidence `directory` → accepted →
     re-validated. A **new** warning appears: the invite still points to the typo address, which is escalated.
   * **Grace**: `gmial.com` → the agent does *not* guess `gmail.com`, because no source confirms the mailbox. If it tried
     DNS as evidence, the engine would reject it. It goes to human review.
5. **Result tabs**:
   * the cleansed list (statuses) and CSV download
   * the review queue
   * the corrections log, including any rejections
   * provenance for Johan: two addresses merged, the truncated phone replaced by the directory value, the source of
     each field
   * excluded: the organiser and the room, plus any internal staff the extraction picked up
6. Optionally, click **Create Gmail draft** and show the draft in Gmail. It is never sent.

## 3. Limitations and next steps (3 min)
Use the final section of the technical summary: SQL/DuckDB for multi-table aggregation, cross-encoder reranking,
evaluating the agent's runs against ground truth, calendar write-back with approval, and production OAuth and secrets.

## Fallbacks
* If Google OAuth fails, switch to Fixture. It is the same scenario, the same tools interface and the same logic.
* If OpenAI is slow, the Evaluation tab and the tests still demonstrate the deterministic parts.
