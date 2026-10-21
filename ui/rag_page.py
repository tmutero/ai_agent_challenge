"""Part 1 UI: chat over the structured knowledge base, with full traceability."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from part1_rag.engine import RagEngine, RagResponse
from part1_rag.loaders import SUPPORTED, SourceFile, build_knowledge_base, sample_files
from ui.errors import show_llm_error

ROOT = Path(__file__).parent.parent
SAMPLE_NAMES = {f.name for f in sample_files()}


# --- per-session document store ------------------------------------------------------
# Each browser session has its own files and index, so on a shared deployment one
# visitor's uploads are never visible to another. Embeddings are cached process-wide
# by content hash, so rebuilding the index after an upload only embeds new chunks.

def _files() -> dict[str, SourceFile]:
    if "rag_files" not in st.session_state:
        st.session_state.rag_files = {f.name: f for f in sample_files()}   # start with the samples
    return st.session_state.rag_files


def get_engine() -> RagEngine:
    files = _files()
    fingerprint = tuple(sorted((n, f.sha256) for n, f in files.items()))
    if st.session_state.get("rag_fingerprint") != fingerprint:
        with st.spinner(f"Indexing {len(files)} document(s): parse -> chunk -> embed -> Qdrant..."):
            kb, reports = build_knowledge_base(list(files.values()))
            try:
                st.session_state.rag_engine = RagEngine(kb=kb)
            except Exception as exc:  # noqa: BLE001 -- show_llm_error re-raises anything that isn't OpenAI
                show_llm_error(exc, "Couldn't index the documents")
                st.stop()   # fingerprint not saved, so the next rerun retries
        st.session_state.rag_reports = reports
        st.session_state.rag_fingerprint = fingerprint
    return st.session_state.rag_engine


def _add_uploads(uploads) -> None:
    files = _files()
    known = {f.sha256: n for n, f in files.items()}
    notes = []
    for up in uploads:
        data = up.getvalue()
        f = SourceFile(up.name, data)
        if f.sha256 in known:
            notes.append(f"'{up.name}' skipped - identical to '{known[f.sha256]}'")
            continue
        if up.name in files:
            notes.append(f"'{up.name}' replaced the previous version")
        files[up.name] = f
        known[f.sha256] = up.name
    st.session_state.rag_upload_notes = notes
    st.session_state.rag_uploader_key = st.session_state.get("rag_uploader_key", 0) + 1   # clear the widget


def _verification_badge(r: RagResponse) -> str:
    if r.refused:
        return ":orange-badge[Refused - not enough evidence]" if r.plan and r.plan.intent != "out_of_scope" \
            else ":gray-badge[Out of scope]"
    n = len(r.fact_checks)
    ok = sum(c.verified for c in r.fact_checks)
    if n == 0:
        return ":gray-badge[No checkable facts]"
    colour = "green" if ok == n and not r.unsupported_numbers else "red"
    badge = f":{colour}-badge[{ok}/{n} facts verified against cited source]"
    if r.unsupported_numbers:
        badge += f" :red-badge[numbers not found in any source: {', '.join(r.unsupported_numbers)}]"
    return badge


def _sources(r: RagResponse) -> None:
    cited = set()
    if r.answer:
        import re
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", r.answer.answer)} | {f.passage for f in r.answer.facts}
    rows = [{"#": i, "source": p.label, "cited": "yes" if i in cited else ""} for i, p in enumerate(r.passages, 1)]
    if rows:
        st.caption("Sources")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _trace(r: RagResponse) -> None:
    with st.expander("How this answer was produced"):
        if r.plan:
            st.markdown("**1. Query transformation** (LLM -> structured plan -> validated against catalog)")
            st.json({
                "standalone_query": r.plan.standalone_query,
                "intent": r.plan.intent,
                "datasets": r.plan.datasets,
                "filters": [f.model_dump() for f in r.plan.filters],
                "sub_queries": r.plan.sub_queries,
                "aggregation": r.plan.aggregation.model_dump() if r.plan.aggregation else None,
                "reasoning": r.plan.reasoning,
            }, expanded=False)
            for note in r.plan_notes:
                st.warning(f"Planner correction: {note}")
        if r.aggregation is not None:
            st.markdown("**2. Deterministic aggregation** (pandas over the full table - not top-k)")
            st.code(r.aggregation.description + (f" = {r.aggregation.value}" if r.aggregation.value is not None else ""))
            st.dataframe(r.aggregation.rows, hide_index=True, width="stretch")
            if r.aggregation.excluded_missing:
                st.caption(f"Excluded (value not reported): {', '.join(r.aggregation.excluded_missing)}")
        elif r.routes or r.retrieved:
            st.markdown("**2. Hybrid retrieval** (dense + BM25, RRF fusion, Qdrant payload pre-filters)")
            st.caption("Routes (one filtered search per dataset): " + ", ".join(str(x) for x in r.routes))
            st.dataframe(pd.DataFrame([{
                "chunk": c.chunk.chunk_id, "dense (cosine)": c.dense_score, "bm25": c.sparse_score,
                "fused (RRF)": c.fused_score,
            } for c in r.retrieved]), hide_index=True, width="stretch")
            st.caption(f"Evidence gate: best dense similarity {r.confidence:.3f} "
                       f"(threshold {get_engine().settings.min_grounding_score})")
        if r.fact_checks:
            st.markdown("**3. Answer verification** (each extracted value must occur in the passage it cites)")
            st.dataframe(pd.DataFrame([{
                "claim": c.fact.claim, "value": c.fact.value, "passage": c.fact.passage,
                "verified": "✅" if c.verified else "❌", "check": c.reason,
            } for c in r.fact_checks]), hide_index=True, width="stretch")
        if r.passages:
            st.markdown("**Context passages sent to the model**")
            for i, p in enumerate(r.passages, 1):
                st.text(f"[{i}] {p.label}\n{p.text}")
        st.caption(f"Timings (ms): {r.timings_ms}")


def _render_assistant(r: RagResponse) -> None:
    st.markdown(r.text)
    st.markdown(_verification_badge(r))
    _sources(r)
    _trace(r)


def _ask(question: str) -> None:
    history = [(m["question"], m["response"].text) for m in st.session_state.rag_turns]
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Planning -> retrieving -> answering -> verifying..."):
                r = get_engine().answer(question, history)
        except Exception as exc:  # noqa: BLE001 -- show_llm_error re-raises anything that isn't OpenAI
            show_llm_error(exc, "Couldn't answer the question")
            return   # not added to the history, so the question can simply be asked again
        _render_assistant(r)
    st.session_state.rag_turns.append({"question": question, "response": r})


def _documents_tab(engine: RagEngine) -> None:
    files = _files()
    st.markdown("Upload structured documents - tables are indexed **one row per chunk** and are also available "
                "to the aggregation engine; prose is indexed per paragraph. Supported: "
                + ", ".join(f"`{x}`" for x in SUPPORTED) + ".")
    uploads = st.file_uploader("Add documents", type=[x.lstrip(".") for x in SUPPORTED], accept_multiple_files=True,
                               key=f"uploader-{st.session_state.get('rag_uploader_key', 0)}")
    if uploads:
        _add_uploads(uploads)
        st.rerun()
    for note in st.session_state.pop("rag_upload_notes", []):
        st.info(note)

    c1, c2 = st.columns(2)
    if c1.button("Load sample documents", disabled=SAMPLE_NAMES <= set(files)):
        files.update({f.name: f for f in sample_files()})
        st.rerun()
    if c2.button("Remove all documents", disabled=not files):
        files.clear()
        st.session_state.rag_turns = []
        st.rerun()

    reports = {r.source: r for r in st.session_state.get("rag_reports", [])}
    st.subheader(f"Loaded documents ({len(files)})")
    for name in sorted(files):
        r = reports.get(name)
        cols = st.columns([4, 5, 1])
        cols[0].markdown(f"**{name}**" + (" _(sample)_" if name in SAMPLE_NAMES else ""))
        if r is None:
            cols[1].caption("pending")
        elif r.error:
            cols[1].error(r.error)
        else:
            parts = [f"{r.records} chunks"]
            if r.datasets:
                parts.append(f"tables: {', '.join(r.datasets)}")
            if r.text_paragraphs:
                parts.append(f"{r.text_paragraphs} paragraphs")
            cols[1].caption(" - ".join(parts))
            for w in r.warnings:
                cols[1].warning(w)
        if cols[2].button("Remove", key=f"rm-{name}"):
            del files[name]
            st.rerun()

    if engine.kb.datasets:
        st.subheader("Extracted tables")
        st.caption("Each row is one chunk in the vector index and one row for the aggregation engine; both carry "
                   "the same `_source` / `_record_ref` / `_chunk_id`.")
        for name, ds in engine.kb.datasets.items():
            with st.expander(f"{name} - {ds.source} - {ds.description}"):
                if ds.filterable:
                    st.caption("Filterable columns: " + ", ".join(ds.filterable))
                st.dataframe(ds.frame, hide_index=True, width="stretch")
    if engine.kb.text_sources:
        st.subheader("Prose")
        for t in engine.kb.text_sources.values():
            with st.expander(f"{t.name} - {t.source} ({t.paragraphs} paragraphs)"):
                for d in engine.kb.documents:
                    if d.metadata.get("category") == t.name:
                        st.text(f"[{d.metadata['record_ref']}] {d.text}")


def render() -> None:
    st.title("Part 1 - RAG over your documents")
    st.caption("Upload CSV, Excel, JSON, PDF, Word or text files - or use the bundled sample documents.")
    st.session_state.setdefault("rag_turns", [])
    engine = get_engine()

    chat_tab, docs_tab, eval_tab = st.tabs(["Chat", "Documents", "Evaluation"])

    with st.sidebar:
        st.subheader("Try a question")
        for q in engine.kb.example_questions():
            if st.button(q, key=f"ex-{q}", width="stretch"):
                st.session_state.rag_pending = q
        if st.button("Clear conversation", type="secondary"):
            st.session_state.rag_turns = []
        st.divider()
        s = engine.settings
        st.caption(f"Index: {engine.indexed_chunks} chunks from {len(engine.kb.sources)} file(s) - "
                   f"{s.embedding_model} + BM25 - RRF(k={s.rrf_k}) - top_k={s.top_k} - "
                   f"grounding threshold {s.min_grounding_score}")

    with chat_tab:
        if engine.kb.empty:
            st.info("No documents loaded. Add some in the **Documents** tab (or load the samples).")
        for turn in st.session_state.rag_turns:
            with st.chat_message("user"):
                st.markdown(turn["question"])
            with st.chat_message("assistant"):
                _render_assistant(turn["response"])
        pending = st.session_state.pop("rag_pending", None)
        typed = st.chat_input("Ask a question about the loaded documents")
        if q := (typed or pending):
            _ask(q)

    with docs_tab:
        _documents_tab(engine)

    with eval_tab:
        st.caption("Measured on the bundled sample documents.")
        path = ROOT / "docs" / "eval_results.md"
        st.markdown(path.read_text() if path.exists() else "Run `python -m part1_rag.evaluate` to generate.")
