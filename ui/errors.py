"""Show OpenAI failures as a readable message instead of a traceback."""

from __future__ import annotations

import logging

import streamlit as st

from common.llm import explain_llm_error

logger = logging.getLogger(__name__)


def show_llm_error(exc: Exception, title: str) -> None:
    """Render ``exc`` if it is an OpenAI failure; re-raise anything else (a real bug should still surface)."""
    message = explain_llm_error(exc)
    if message is None:
        raise exc
    logger.warning("%s: %s", title, exc)
    st.error(message, title=title, icon=":material/error:")
