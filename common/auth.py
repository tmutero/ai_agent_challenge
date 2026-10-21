"""Single-user login gate. Credentials come from APP_USERNAME / APP_PASSWORD (env or Streamlit secrets)."""

from __future__ import annotations

import hmac

import streamlit as st

from common.llm import secret


def _matches(given: str, expected: str) -> bool:
    return hmac.compare_digest(given.encode(), expected.encode())


def require_login() -> None:
    """Show a login form and stop the page until the user has signed in."""
    if st.session_state.get("authenticated"):
        with st.sidebar:
            st.caption(f"Signed in as **{st.session_state.auth_user}**")
            if st.button("Sign out", icon=":material/logout:"):
                st.session_state.clear()
                st.rerun()
        return

    username, password = secret("APP_USERNAME"), secret("APP_PASSWORD")
    if not username or not password:   # fail closed: never show the app without a configured login
        st.error("Set APP_USERNAME and APP_PASSWORD in .env or Streamlit secrets.", title="Login is not configured")
        st.stop()

    _, centre, _ = st.columns([1, 2, 1])
    with centre:
        st.title("Sign in")
        with st.form("login"):
            user = st.text_input("Username", autocomplete="username")
            pwd = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
        if submitted:
            # Compare both fields every time, so a wrong username fails as fast as a wrong password.
            if _matches(user.strip(), username) & _matches(pwd, password):
                st.session_state.authenticated = True
                st.session_state.auth_user = user.strip()
                st.rerun()
            st.error("Incorrect username or password.")
    st.stop()
