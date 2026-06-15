"""
Streamlit front-end: Chat + Monitoring Dashboard.

ChatGPT-style presentation layer. All pipeline logic, guardrail processing,
logging, and dashboard functionality are unchanged — only the visuals differ.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import PipelineConfig, DB_PATH
from src.pipeline import GuardrailPipeline
from src.logging_db import EventLog


st.set_page_config(
    page_title="ChatGuard",
    page_icon="💬",
    layout="centered",
)


# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* --- hide Streamlit chrome --- */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* --- global font stack --- */
html, body, [class*="css"] {
    font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue",
                 "Liberation Sans", Arial, sans-serif;
    font-size: 15px;
}

/* --- page background --- */
.stApp {
    background-color: #f7f7f8;
}

/* --- content column: constrained width, room for pinned input --- */
.main .block-container {
    max-width: 740px;
    padding-top: 1.75rem;
    padding-bottom: 6.5rem;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
}

/* --- chat message wrapper: no extra box or shadow --- */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.3rem 0 !important;
    gap: 0.6rem !important;
}

/* --- avatars: hide them; visual identity comes from layout/color --- */
[data-testid^="stChatMessageAvatar"] {
    display: none !important;
}

/* --- message body typography --- */
[data-testid="stChatMessageContent"] {
    font-size: 0.95rem;
    line-height: 1.72;
    color: #1c1c1e;
}

/* --- assistant message: white card --- */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"]) {
    /* base reset already applied above */
}

/* Target user messages via the known Streamlit class suffix.
   Falls back gracefully on versions where the selector doesn't match. */
[class*="stChatMessage-user"] [data-testid="stChatMessageContent"],
div[data-testid="stChatMessage"].user [data-testid="stChatMessageContent"] {
    background: #ededee;
    border-radius: 14px;
    padding: 0.6rem 1rem;
    display: inline-block;
    max-width: 90%;
}

/* --- expander: minimal border, muted label --- */
[data-testid="stExpander"] {
    border: 1px solid #e4e4e7 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    box-shadow: none !important;
    margin-top: 0.4rem;
}
[data-testid="stExpander"] summary {
    font-size: 0.78rem !important;
    color: #999 !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
}
[data-testid="stExpander"] summary:hover {
    color: #555 !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 0.5rem 0.75rem 0.75rem;
}

/* --- caption text --- */
[data-testid="stCaptionContainer"] p {
    color: #a0a0a8 !important;
    font-size: 0.73rem !important;
    margin-top: 0.2rem;
}

/* --- chat input: rounded, clean --- */
[data-testid="stChatInput"] {
    background: #fff;
    border-radius: 14px;
    border: 1px solid #d4d4d8;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stChatInput"] textarea {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    color: #1c1c1e !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.65rem 0.9rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #b0b0b8 !important;
}

/* --- tab bar: minimal underline style --- */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #e4e4e7 !important;
    gap: 0 !important;
    margin-bottom: 0.5rem;
}
button[data-baseweb="tab"] {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #888 !important;
    padding: 0.45rem 1.1rem !important;
    border-radius: 0 !important;
    background: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1c1c1e !important;
    border-bottom: 2px solid #1c1c1e !important;
    background: transparent !important;
}

/* --- dashboard metric cards --- */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}

/* --- blocked-message notice: amber left-bar, no harsh red --- */
.blocked-notice {
    background: #fffbf0;
    border-left: 3px solid #d4900a;
    border-radius: 0 8px 8px 0;
    padding: 0.55rem 0.9rem;
    font-size: 0.9rem;
    color: #5c4300;
    line-height: 1.6;
    margin-bottom: 0.2rem;
}
</style>
""", unsafe_allow_html=True)


# ── cached resources ───────────────────────────────────────────────────────

@st.cache_resource
def get_pipeline():
    return GuardrailPipeline(PipelineConfig())


def get_log():
    return EventLog(DB_PATH)


# ── guardrail trace table (unchanged logic) ────────────────────────────────

def render_guardrail_trace(results, title):
    if not results:
        return
    st.caption(title)
    df = pd.DataFrame([
        {
            "guardrail": r["name"],
            "triggered": "yes" if r["triggered"] else "no",
            "score": r["score"],
            "ms": r["elapsed_ms"],
            "reason": r["reason"],
        }
        for r in results
    ])
    st.dataframe(df, hide_index=True, use_container_width=True)


# ── tabs ───────────────────────────────────────────────────────────────────

chat_tab, dash_tab = st.tabs(["Chat", "Dashboard"])


# ══════════════════════════════════════════════════════════════════════════
# CHAT TAB
# ══════════════════════════════════════════════════════════════════════════

with chat_tab:
    # Minimal, muted header — replaces the big title + description
    st.markdown(
        "<p style='color:#c0c0c8;font-size:0.73rem;text-align:center;"
        "letter-spacing:0.08em;margin-top:0.25rem;margin-bottom:1.5rem;'>"
        "HINGLISH GUARDRAILS</p>",
        unsafe_allow_html=True,
    )

    # ── session state ──────────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []   # list of (role, content)
    if "conv_id" not in st.session_state:
        st.session_state.conv_id = None

    # ── replay conversation history ────────────────────────────────────────
    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.write(content)

    # ── new message ────────────────────────────────────────────────────────
    user_msg = st.chat_input("Message…")
    if user_msg:
        with st.chat_message("user"):
            st.write(user_msg)

        pipeline = get_pipeline()

        # build history for the LLM (same logic as before)
        hist = [
            {"role": r, "content": c}
            for r, c in st.session_state.history
            if r in ("user", "assistant")
        ]

        with st.spinner("Thinking…"):
            result = pipeline.process(
                user_msg,
                conversation_id=st.session_state.conv_id,
                history=hist,
            )
        st.session_state.conv_id = result["conversation"]

        with st.chat_message("assistant"):
            if result["blocked"]:
                # subtle amber notice — no harsh red error box
                st.markdown(
                    f'<div class="blocked-notice">🚫 {result["response"]}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"blocked by: **{result['blocked_by']}**")
            else:
                st.write(result["response"])

            # guardrail trace tucked inside a collapsed expander
            with st.expander("details", expanded=False):
                st.caption(
                    f"language: **{result['language']}**  ·  "
                    f"latency: **{result['total_ms']} ms**"
                )
                render_guardrail_trace(result["input_results"], "input guardrails")
                render_guardrail_trace(result["output_results"], "output guardrails")

        # ── persist both turns in session state ────────────────────────────
        st.session_state.history.append(("user", user_msg))
        st.session_state.history.append(("assistant", result["response"]))


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD TAB  — logic entirely unchanged, inherits new font/colors
# ══════════════════════════════════════════════════════════════════════════

with dash_tab:
    st.title("Monitoring Dashboard")
    log = get_log()
    stats = log.stats()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total messages", stats["total"])
    c2.metric("Blocked", stats["blocked"])
    c3.metric("Allowed", stats["allowed"])

    colA, colB = st.columns(2)
    with colA:
        st.subheader("By language")
        if stats["by_language"]:
            st.bar_chart(pd.Series(stats["by_language"], name="count"))
        else:
            st.info("No data yet. Send a few messages in the Chat tab.")
    with colB:
        st.subheader("Blocks by guardrail")
        if stats["by_guardrail"]:
            st.bar_chart(pd.Series(stats["by_guardrail"], name="count"))
        else:
            st.info("No blocks recorded yet.")

    st.subheader("Recent events")
    rows = log.recent(limit=50)
    if rows:
        table = pd.DataFrame([
            {
                "id": r["id"],
                "lang": r["language"],
                "action": r["final_action"],
                "blocked_by": r["blocked_by"] or "-",
                "input": (r["user_input"] or "")[:60],
                "ms": round(r["total_ms"], 1),
            }
            for r in rows
        ])
        st.dataframe(table, hide_index=True, use_container_width=True)
    else:
        st.info("No events logged yet.")
