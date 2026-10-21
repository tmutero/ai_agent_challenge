"""Streamlit entry point: `streamlit run app.py`."""

import logging
import warnings

import streamlit as st

from common.auth import require_login

warnings.filterwarnings("ignore", message=".*Payload indexes have no effect.*")
logging.getLogger("httpx").setLevel(logging.WARNING)

st.set_page_config(page_title="Kestrel AI Agent Challenge", page_icon="🦅", layout="wide")

require_login()


def rag():
    from ui import rag_page
    rag_page.render()


def agent():
    from ui import agent_page
    agent_page.render()


def about():
    st.title("About this submission")
    st.markdown(
        "My response to the AI Agent Technical Challenge: one Streamlit app covering both parts, built on a shared "
        "Python core. I built it in code rather than a no-code tool so that every step can be inspected, tested "
        "and measured."
    )
    st.info(
        "The model handles the language work: rewriting questions, reading messy emails and deciding what to "
        "check next. Values, joins, rules and accepted corrections are handled by deterministic code that is "
        "unit-tested.",
        title="Design principle: the LLM proposes, code verifies",
        icon=":material/verified:",
    )

    part1, part2 = st.columns(2)
    with part1.container(border=True, height="stretch"):
        st.subheader(":material/description: Part 1 - RAG over documents")
        st.markdown(
            "Ask questions about uploaded CSV, Excel, JSON, PDF, Word or text files and get cited answers.\n\n"
            "- **Ingestion**: one chunk per record (table row, JSON object or paragraph), never a blind character "
            "split. Key, title and filter columns are inferred from the data, so no dataset is hard-coded.\n"
            "- **Retrieval**: hybrid search (OpenAI embeddings + BM25, RRF fusion) in Qdrant, with metadata "
            "pre-filters.\n"
            "- **Query planning**: follow-up questions are rewritten as standalone queries and routed to the right "
            "dataset. Ranking and counting questions run in pandas over the full table instead of top-k search.\n"
            "- **Grounding**: low-confidence retrieval is refused, and each value in the answer is checked against "
            "the passage it cites."
        )
    with part2.container(border=True, height="stretch"):
        st.subheader(":material/smart_toy: Part 2 - Attendee reconciliation agent")
        st.markdown(
            "*\"Prepare the verified attendee list for the Q4 Investor Roadshow in Cape Town.\"*\n\n"
            "- **Tools**: Google Calendar, Gmail, a public partner directory and DNS lookups. Each runs live or "
            "on an offline copy of the same scenario.\n"
            "- **Data handling**: phone and name normalisation, de-duplication, conflict resolution (latest RSVP "
            "wins) and validation rules such as domain typos and missing details.\n"
            "- **Self-correction**: a fix is only accepted when it comes with evidence, such as a directory row or "
            "email that contains the new value. Anything unproven is flagged for a person.\n"
            "- **Output**: a cleansed table with per-field provenance, a review queue, a corrections log and a CSV "
            "export. Email drafts are never sent automatically."
        )

    st.header("How to review it")
    st.markdown(
        "- Under each Part 1 answer, open **How this answer was produced** to see the plan, the retrieval scores, "
        "the passages and the fact checks.\n"
        "- In Part 2, the **Full trace** tab shows every reasoning step and tool call.\n"
        "- `pytest` runs 30 tests offline, with no API key needed."
    )

    st.subheader("Retrieval evaluation (20 labelled questions, k=5)")
    st.table(
        {
            "Configuration": ["Dense only", "Hybrid (dense + BM25)", "Hybrid + query planner (shipped)"],
            "Recall": [0.85, 0.95, 1.00],
            "MRR": [0.80, 0.86, 1.00],
        },
        alt="Retrieval recall and MRR for three pipeline configurations",
    )
    st.caption("These results drove the configuration choices: BM25 helps with codes and exact terms, and "
               "planning and routing give the largest gain.")

    st.header("With more time")
    st.markdown(
        "- Text-to-SQL over DuckDB for joins and range filters across tables.\n"
        "- A cross-encoder reranker and an end-to-end answer-correctness evaluation.\n"
        "- Repeated agent runs scored against the scenario's ground truth to measure variance.\n"
        "- Calendar write-back behind a human approval step, plus secrets management and PII redaction in traces."
    )


st.navigation([
    st.Page(rag, title="Part 1 - RAG", icon="📚", url_path="rag", default=True),
    st.Page(agent, title="Part 2 - Attendee agent", icon="🤖", url_path="agent"),
    st.Page(about, title="About", icon="📝", url_path="about"),
]).run()
