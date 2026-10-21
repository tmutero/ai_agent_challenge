"""Part 2 UI: run the attendee agent, watch its reasoning live, inspect the verified output."""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

from part2_agent.agent import DEFAULT_GOAL, AgentRun, AttendeeAgent, Step
from part2_agent.connectors import CREDENTIALS_PATH, TOKEN_PATH, make_toolset
from ui.errors import show_llm_error

STATUS_ICON = {"verified": "✅ verified", "flagged": "⚠️ flagged", "human review": "👤 human review",
               "needs fix": "❌ needs fix"}
ICON = {"reasoning": "💭", "call": "🔧", "result": "📥", "error": "⛔", "final": "🏁"}


def _google_available() -> bool:
    from common.llm import secret
    return TOKEN_PATH.exists() or bool(secret("GOOGLE_TOKEN_JSON"))


def _render_step(container, step: Step) -> None:
    with container:
        if step.kind == "reasoning":
            st.markdown(f"{ICON['reasoning']} _{step.payload}_")
        elif step.kind == "call":
            args = json.dumps(step.payload, ensure_ascii=False)
            st.markdown(f"{ICON['call']} **{step.name}** `{args[:300]}`")
        elif step.kind == "result":
            with st.expander(f"{ICON['result']} {step.name} result ({step.ms} ms)"):
                st.json(step.payload, expanded=False) if not isinstance(step.payload, str) else st.write(step.payload)
        elif step.kind == "error":
            st.error(f"{step.name}: {step.payload}")


def _run(mode: str, goal: str, max_turns: int) -> AgentRun | None:
    tools = make_toolset(mode)
    status = st.status("Agent running...", expanded=True)
    log = status.container(height=480)
    agent = AttendeeAgent(tools, max_turns=max_turns, on_step=lambda s: _render_step(log, s))
    try:
        run = agent.run(goal)
    except Exception as exc:  # noqa: BLE001 -- show_llm_error re-raises anything that isn't OpenAI
        status.update(label="Agent stopped - OpenAI request failed", state="error", expanded=True)
        show_llm_error(exc, "The agent couldn't finish")
        return None   # the steps so far stay visible in the log above
    status.update(label=f"Agent finished - {sum(s.kind == 'call' for s in run.steps)} tool calls, "
                        f"{run.tokens:,} tokens", state="complete" if run.finished else "error", expanded=False)
    st.session_state.agent_tools = tools
    return run


def _results(run: AgentRun) -> None:
    ws = run.workspace
    rows, issues = ws.table()
    attendees, _ = ws.build()
    df = pd.DataFrame(rows)
    df["status"] = df["status"].map(STATUS_ICON).fillna(df["status"])

    st.subheader(f"Result - {run.event_summary or 'event'}")
    if run.summary:
        st.info(run.summary)
    accepted = sum(c.accepted for c in ws.corrections)
    cols = st.columns(6)
    cols[0].metric("Attendees", len(rows))
    cols[1].metric("Verified", sum(r["status"] == "verified" for r in rows))
    cols[2].metric("Human review", sum(r["status"] == "human review" for r in rows))
    cols[3].metric("Needs fix / flagged", sum(r["status"] in ("needs fix", "flagged") for r in rows))
    cols[4].metric("Corrections", f"{accepted} ✓ / {len(ws.corrections) - accepted} ✗")
    cols[5].metric("Excluded", len(ws.excluded) + len(ws.internal_excluded))

    t_list, t_issues, t_corr, t_prov, t_excl, t_trace = st.tabs(
        ["Cleansed attendee list", "Issues & review queue", "Corrections log", "Provenance", "Excluded", "Full trace"])

    with t_list:
        st.dataframe(df.drop(columns=["key"]), hide_index=True, width="stretch", column_config={
            "on_invite": st.column_config.CheckboxColumn("on invite"),
            "flags": st.column_config.TextColumn("flags", width="large")})
        st.download_button("Download CSV", df.drop(columns=["key"]).to_csv(index=False).encode(),
                           file_name="roadshow_attendees_verified.csv", mime="text/csv")
        if st.button("Create Gmail draft with this list"):
            body = "Verified attendee list\n\n" + df.drop(columns=["key"]).to_string(index=False) + \
                   "\n\nOpen issues:\n" + "\n".join(f"- {i.person}: {i.message}" for i in issues if i.severity != "info")
            organiser = next((a["email"] for a in (ws.event or {}).get("attendees", []) if a.get("organizer")), "me")
            tools = st.session_state.get("agent_tools")
            draft_id = tools.mail.create_draft(organiser, f"Attendee list - {run.event_summary}", body)
            st.success(f"Draft created ({draft_id}) for {organiser} - not sent.")

    with t_issues:
        idf = pd.DataFrame([asdict(i) for i in issues])
        if not idf.empty:
            idf = idf.sort_values(["status", "severity"], key=lambda s: s.map(
                {"open": 0, "human_review": 1, "error": 0, "warning": 1, "info": 2}).fillna(3))
            st.dataframe(idf[["person", "severity", "code", "message", "suggestion", "status", "review_note"]],
                         hide_index=True, width="stretch")

    with t_corr:
        if ws.corrections:
            st.dataframe(pd.DataFrame([asdict(c) for c in ws.corrections]), hide_index=True, width="stretch")
            st.caption("A correction is accepted only if the new value literally appears in the cited evidence "
                       "(directory entry or a read email). The row is then re-validated by the same rules.")
        else:
            st.write("No corrections proposed.")
        if ws.extraction_rejections:
            st.markdown("**Extraction values rejected by the grounding check**")
            st.write(ws.extraction_rejections)

    with t_prov:
        for a in attendees:
            with st.expander(f"{a.name} - {a.email}"):
                st.markdown("**Field sources**")
                st.json(a.sources, expanded=True)
                if a.notes:
                    st.markdown("**Merges, conflicts and normalisations**")
                    for n in a.notes:
                        st.markdown(f"- {n}")
                obs = [asdict(o) for o in ws.observations if o.obs_id in a.observation_ids]
                st.markdown("**Raw observations**")
                st.dataframe(pd.DataFrame(obs), hide_index=True, width="stretch")

    with t_excl:
        st.dataframe(pd.DataFrame(ws.excluded + ws.internal_excluded), hide_index=True, width="stretch")
        if ws.irrelevant_messages:
            st.caption(f"Emails judged unrelated to the event: {', '.join(ws.irrelevant_messages)}")
        d = ws.web.directory()
        st.caption(f"Partner directory: {d.url} (fetched via {d.fetched_via}; {len(d.people)} representatives, "
                   f"{len(d.firms)} firms). {' '.join(d.notes)}")

    with t_trace:
        for s in run.steps:
            _render_step(st.container(), s)


def render() -> None:
    st.title("Part 2 - Attendee reconciliation agent")
    st.caption("Calendar + Email + Web: fetch the event and RSVP/registration emails, extract and normalise people, "
               "verify against a public partner directory and DNS, correct only with evidence, flag the rest.")

    with st.sidebar:
        st.subheader("Agent settings")
        google_ok = _google_available()
        options = ["Fixture (offline scenario)"] + (["Google (live Gmail + Calendar)"] if google_ok else [])
        choice = st.radio("Data source", options, index=len(options) - 1)
        if not google_ok:
            st.caption("Live Google mode appears once `token.json` exists "
                       f"({'credentials.json found' if CREDENTIALS_PATH.exists() else 'see README: Google setup'}).")
        max_turns = st.slider("Max agent turns", 10, 40, 30)
    mode = "google" if choice.startswith("Google") else "fixture"

    goal = st.text_area("Goal", DEFAULT_GOAL, height=90)
    if st.button("Run agent", type="primary"):
        st.session_state.agent_run = _run(mode, goal, max_turns)
    run = st.session_state.get("agent_run")
    if run:
        _results(run)
    else:
        with st.expander("What the agent will do", expanded=True):
            st.markdown(
                "1. **Calendar** - find the event, load attendees (organiser & rooms excluded)\n"
                "2. **Email** - search RSVPs / apologies / registration lists; LLM extraction with a grounding check\n"
                "3. **Consolidate** - normalise (E.164 phones, names, firms), de-duplicate, resolve RSVP conflicts\n"
                "4. **Verify (web)** - scrape the public partner directory over HTTP, DNS MX-check email domains\n"
                "5. **Self-correct** - investigate each issue; apply corrections only with cited evidence; re-validate\n"
                "6. **Escalate** - anything unverifiable goes to a human review queue\n"
                "7. **Output** - cleansed table, CSV export, optional Gmail draft (never auto-sent)")
