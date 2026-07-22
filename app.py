"""
Streamlit front-end: Chat + Monitoring Dashboard.

Dark-mode glassmorphism presentation layer. ALL pipeline logic, guardrail
processing, session-state history threading, conversation_id management, and
SQLite logging are unchanged — only the visual presentation differs. Nothing
under src/ is touched by this file.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import html
import io
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import PipelineConfig, DB_PATH, OLLAMA_MODEL
from src.pipeline import GuardrailPipeline
from src.logging_db import EventLog
# analyze_logs is a top-level helper module (not under src/); reused for the
# dashboard stats, event fetch, and CSV export so the SQL lives in one place.
from analyze_logs import fetch_stats, fetch_events, flatten_results


st.set_page_config(
    page_title="Hinglish Guardrails",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════
# Design system
# ═══════════════════════════════════════════════════════════════════════════

# Accent palette
BLUE    = "#3b82f6"
AMBER   = "#f59e0b"
EMERALD = "#10b981"
ROSE    = "#ef4444"
PURPLE  = "#a855f7"

# guardrail identity → (accent color, display label)
GUARDRAIL_STYLE: dict[str, tuple[str, str]] = {
    "injection":     (AMBER,   "INJECTION"),
    "toxicity":      (ROSE,    "TOXICITY"),
    "jailbreak":     (PURPLE,  "JAILBREAK"),
    "pii":           (BLUE,    "PII"),
    "output_filter": (AMBER,   "OUTPUT FILTER"),
}

# language → display label
LANG_LABEL = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish", "unknown": "Unknown"}


def guardrail_style(blocked_by: str | None) -> tuple[str, str]:
    """Map a blocked_by value (possibly 'output_filter/toxic') to (color, label)."""
    if not blocked_by:
        return (ROSE, "BLOCKED")
    base = blocked_by.split("/")[0]
    color, label = GUARDRAIL_STYLE.get(base, (ROSE, base.upper()))
    if "/" in blocked_by:
        sub = blocked_by.split("/", 1)[1].replace("_", " ").upper()
        label = f"{label} · {sub}"
    return color, label


def severity_color(score: float) -> str:
    """Confidence-bar color: higher score = more severe."""
    if score >= 0.80:
        return ROSE
    if score >= 0.60:
        return AMBER
    return BLUE


BASE_CSS = f"""
<style>
/* ---- hide Streamlit chrome ---- */
#MainMenu {{ visibility: hidden; }}
footer    {{ visibility: hidden; }}
header    {{ visibility: hidden; }}
[data-testid="stStatusWidget"] {{ display: none; }}

/* ---- global font + page background ---- */
html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}
.stApp {{
    background: radial-gradient(1200px 600px at 50% -10%, #14142a 0%, #0a0a0f 55%) fixed;
    color: #e5e7eb;
}}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {{
    background: rgba(12, 12, 22, 0.92);
    border-right: 1px solid rgba(255,255,255,0.06);
}}
[data-testid="stSidebar"] * {{ color: #d1d5db; }}
[data-testid="stSidebar"] .sidebar-logo {{
    font-size: 1.15rem; font-weight: 700; letter-spacing: 0.01em;
    color: #f9fafb; padding: 0.2rem 0 1rem 0;
}}

/* ---- radio nav in sidebar ---- */
[data-testid="stSidebar"] [role="radiogroup"] label {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.4rem;
    transition: all 0.15s ease;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    background: rgba(59,130,246,0.12);
    border-color: rgba(59,130,246,0.4);
}}

/* ---- generic card surface (glassmorphism) ---- */
.glass {{
    background: rgba(20, 20, 35, 0.85);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}}

/* ---- headings / text default ---- */
h1, h2, h3, h4, p, span, label, div {{ color: #e5e7eb; }}

/* ---- expander: subtle glass ---- */
[data-testid="stExpander"] {{
    background: rgba(20,20,35,0.6) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}}
[data-testid="stExpander"] summary {{
    font-size: 0.8rem !important;
    color: #9ca3af !important;
    font-weight: 500 !important;
}}
[data-testid="stExpander"] summary:hover {{ color: #d1d5db !important; }}

/* ---- chat input: rounded glass ---- */
[data-testid="stChatInput"] {{
    background: rgba(20,20,35,0.9);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
}}
[data-testid="stChatInput"] textarea {{
    color: #f3f4f6 !important;
    background: transparent !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{ color: #6b7280 !important; }}

/* ---- chat message bubbles ---- */
[data-testid="stChatMessage"] {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.15rem 0 !important;
}}
[data-testid^="stChatMessageAvatar"] {{ display: none !important; }}
[data-testid="stChatMessageContent"] p {{ color: #e8eaed; }}

/* assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"]) [data-testid="stChatMessageContent"] {{
    background: rgba(30, 30, 48, 0.85);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px 20px 20px 6px;
    padding: 0.7rem 1.05rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
}}
/* user bubble (right-aligned, blue) */
.stChatMessage.st-user [data-testid="stChatMessageContent"],
[class*="stChatMessage"]:has([data-testid="stChatMessageAvatar-user"]) [data-testid="stChatMessageContent"] {{
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border-radius: 20px 20px 6px 20px !important;
    margin-left: auto;
}}
[data-testid="stChatMessageAvatar-user"] ~ [data-testid="stChatMessageContent"] p {{ color: #ffffff; }}

/* ---- download button ---- */
[data-testid="stDownloadButton"] button {{
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.5);
    color: #93c5fd;
    border-radius: 10px;
    font-weight: 600;
}}
[data-testid="stDownloadButton"] button:hover {{
    background: rgba(59,130,246,0.28);
    border-color: {BLUE};
    color: #fff;
}}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Cached resources (pipeline + log wiring UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_pipeline():
    return GuardrailPipeline(PipelineConfig())


def get_log():
    return EventLog(DB_PATH)


def active_guardrail_count(cfg: PipelineConfig) -> int:
    flags = [
        cfg.enable_toxicity, cfg.enable_pii, cfg.enable_injection,
        cfg.enable_jailbreak, cfg.enable_output_filter, cfg.enable_hallucination,
    ]
    return sum(1 for f in flags if f)


# ═══════════════════════════════════════════════════════════════════════════
# HTML render helpers
# ═══════════════════════════════════════════════════════════════════════════

def badge(text: str, color: str) -> str:
    return (
        f"<span style='display:inline-block;background:{color}22;color:{color};"
        f"border:1px solid {color}55;border-radius:999px;padding:2px 10px;"
        f"font-size:0.7rem;font-weight:700;letter-spacing:0.04em;'>{html.escape(text)}</span>"
    )


def find_score(results: list[dict], blocked_by: str | None) -> float:
    """Recover the firing guardrail's score from the result list."""
    if not blocked_by:
        return 0.0
    base = blocked_by.split("/")[0]
    for r in results:
        if r.get("name") == base:
            return float(r.get("score", 0.0))
    return 0.0


def render_blocked_card(result: dict) -> None:
    """Polished 'Security Alert' card for a blocked message."""
    blocked_by = result["blocked_by"]
    color, label = guardrail_style(blocked_by)
    lang = result.get("language", "unknown")
    all_results = (result.get("input_results", []) or []) + (result.get("output_results", []) or [])
    score = find_score(all_results, blocked_by)
    bar_color = severity_color(score)
    reason = ""
    base = (blocked_by or "").split("/")[0]
    for r in all_results:
        if r.get("name") == base:
            reason = r.get("reason", "")
            break
    if not reason:
        reason = result.get("response", "")

    card = f"""
    <div class="glass" style="border-left:4px solid {color};padding:1.1rem 1.25rem;margin:0.35rem 0;">
      <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.55rem;">
        <span style="font-size:1.25rem;filter:drop-shadow(0 0 6px {color}88);">🛡️</span>
        <span style="font-size:1.02rem;font-weight:700;color:#f9fafb;">Message Blocked</span>
        <span style="margin-left:auto;">{badge(label, color)}</span>
      </div>
      <div style="color:#cbd5e1;font-size:0.9rem;line-height:1.6;margin-bottom:0.8rem;">
        {html.escape(reason)}
      </div>
      <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.5rem;">
        <span style="font-size:0.72rem;color:#94a3b8;min-width:78px;">Confidence</span>
        <div style="flex:1;height:8px;background:rgba(255,255,255,0.07);border-radius:999px;overflow:hidden;">
          <div style="width:{score*100:.0f}%;height:100%;background:{bar_color};
               box-shadow:0 0 10px {bar_color}99;border-radius:999px;"></div>
        </div>
        <span style="font-size:0.78rem;font-weight:700;color:{bar_color};min-width:42px;text-align:right;">{score:.2f}</span>
      </div>
      <div style="margin-top:0.6rem;">
        {badge("LANG · " + LANG_LABEL.get(lang, lang).upper(), BLUE)}
      </div>
    </div>
    """
    st.markdown(card, unsafe_allow_html=True)


def render_guardrail_trace(results: list[dict], title: str) -> None:
    """Compact HTML trace table of the guardrails that ran."""
    if not results:
        return
    rows = ""
    for r in results:
        fired = r.get("triggered", False)
        dot = EMERALD if not fired else ROSE
        state = "fired" if fired else "clear"
        rows += (
            f"<tr>"
            f"<td style='padding:5px 10px;color:#e5e7eb;'>{html.escape(str(r.get('name','')))}</td>"
            f"<td style='padding:5px 10px;'><span style='color:{dot};font-weight:600;'>● {state}</span></td>"
            f"<td style='padding:5px 10px;color:#cbd5e1;text-align:right;'>{r.get('score',0):.3f}</td>"
            f"<td style='padding:5px 10px;color:#94a3b8;text-align:right;'>{r.get('elapsed_ms',0):.1f} ms</td>"
            f"<td style='padding:5px 10px;color:#94a3b8;'>{html.escape(str(r.get('reason','') or '')[:70])}</td>"
            f"</tr>"
        )
    st.markdown(
        f"<div style='font-size:0.72rem;color:#94a3b8;text-transform:uppercase;"
        f"letter-spacing:0.06em;margin:0.4rem 0 0.25rem;'>{html.escape(title)}</div>"
        f"<table style='width:100%;border-collapse:collapse;font-size:0.8rem;'>"
        f"<thead><tr style='color:#6b7280;font-size:0.68rem;text-transform:uppercase;'>"
        f"<th style='text-align:left;padding:0 10px;'>Guardrail</th>"
        f"<th style='text-align:left;padding:0 10px;'>State</th>"
        f"<th style='text-align:right;padding:0 10px;'>Score</th>"
        f"<th style='text-align:right;padding:0 10px;'>Time</th>"
        f"<th style='text-align:left;padding:0 10px;'>Reason</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>",
        unsafe_allow_html=True,
    )


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e5e7eb", family="-apple-system, Segoe UI, sans-serif"),
    margin=dict(l=10, r=10, t=30, b=10),
)


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("<div class='sidebar-logo'>🛡️ Hinglish Guardrails</div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["💬 Chat", "📊 Dashboard"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:1rem 0;'>", unsafe_allow_html=True)

    cfg = PipelineConfig()
    st.markdown(
        "<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;"
        "color:#6b7280;margin-bottom:0.5rem;'>System Info</div>"
        f"<div style='font-size:0.8rem;line-height:1.9;color:#9ca3af;'>"
        f"<b style='color:#d1d5db;'>LLM</b> · {html.escape(OLLAMA_MODEL)}<br>"
        f"<b style='color:#d1d5db;'>Active guardrails</b> · {active_guardrail_count(cfg)}<br>"
        f"<b style='color:#d1d5db;'>Database</b> · {html.escape(DB_PATH)}"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='position:fixed;bottom:1rem;font-size:0.68rem;color:#4b5563;'>"
        # TODO: replace with the actual team member names for your submission
        "Team · Member One · Member Two · Member Three"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — CHAT
# ═══════════════════════════════════════════════════════════════════════════

def render_chat_page() -> None:
    # Constrain the chat column to a centered ~720px band (chat page only).
    st.markdown(
        """
        <style>
        .main .block-container { max-width: 760px; padding-top: 1rem; padding-bottom: 7rem; }
        [data-testid="stChatInput"] { max-width: 760px; margin: 0 auto; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Slim header bar: project name (left) + shield-active status (right)
    st.markdown(
        f"""
        <div class="glass" style="display:flex;align-items:center;padding:0.7rem 1.1rem;
             margin-bottom:1.1rem;">
          <span style="font-weight:700;font-size:1.02rem;color:#f9fafb;">Hinglish Guardrails</span>
          <span style="margin-left:auto;display:flex;align-items:center;gap:0.45rem;
               font-size:0.78rem;color:#a7f3d0;">
            <span style="width:9px;height:9px;border-radius:50%;background:{EMERALD};
                 box-shadow:0 0 8px {EMERALD};display:inline-block;
                 animation:pulse 2s infinite;"></span>
            Shield active
          </span>
        </div>
        <style>@keyframes pulse {{0%,100%{{opacity:1}}50%{{opacity:0.45}}}}</style>
        """,
        unsafe_allow_html=True,
    )

    # ---- session state (UNCHANGED threading) ----
    if "history" not in st.session_state:
        st.session_state.history = []   # list of (role, content)
    if "conv_id" not in st.session_state:
        st.session_state.conv_id = None
    if "traces" not in st.session_state:
        # parallel to assistant turns: stores the full result dict for replay
        st.session_state.traces = {}

    # ---- replay conversation history ----
    for idx, (role, content) in enumerate(st.session_state.history):
        with st.chat_message(role):
            trace = st.session_state.traces.get(idx)
            if trace and trace.get("blocked"):
                render_blocked_card(trace)
                with st.expander("🛡️ Guardrail trace", expanded=False):
                    render_guardrail_trace(trace.get("input_results", []), "input guardrails")
                    render_guardrail_trace(trace.get("output_results", []), "output guardrails")
            else:
                st.write(content)
                if trace:
                    with st.expander("🛡️ Guardrail trace", expanded=False):
                        st.markdown(
                            f"<span style='font-size:0.75rem;color:#94a3b8;'>"
                            f"language <b>{LANG_LABEL.get(trace.get('language',''), trace.get('language',''))}</b> · "
                            f"latency <b>{trace.get('total_ms','?')} ms</b></span>",
                            unsafe_allow_html=True,
                        )
                        render_guardrail_trace(trace.get("input_results", []), "input guardrails")
                        render_guardrail_trace(trace.get("output_results", []), "output guardrails")

    # ---- new message (pipeline call UNCHANGED) ----
    user_msg = st.chat_input("Type a message in English, Hinglish, or Hindi...")
    if user_msg:
        with st.chat_message("user"):
            st.write(user_msg)

        pipeline = get_pipeline()
        hist = [
            {"role": r, "content": c}
            for r, c in st.session_state.history
            if r in ("user", "assistant")
        ]

        with st.spinner("Running guardrails…"):
            result = pipeline.process(
                user_msg,
                conversation_id=st.session_state.conv_id,
                history=hist,
            )
        st.session_state.conv_id = result["conversation"]

        with st.chat_message("assistant"):
            if result["blocked"]:
                render_blocked_card(result)
                with st.expander("🛡️ Guardrail trace", expanded=False):
                    render_guardrail_trace(result["input_results"], "input guardrails")
                    render_guardrail_trace(result["output_results"], "output guardrails")
            else:
                st.write(result["response"])
                with st.expander("🛡️ Guardrail trace", expanded=False):
                    st.markdown(
                        f"<span style='font-size:0.75rem;color:#94a3b8;'>"
                        f"language <b>{LANG_LABEL.get(result['language'], result['language'])}</b> · "
                        f"latency <b>{result['total_ms']} ms</b></span>",
                        unsafe_allow_html=True,
                    )
                    render_guardrail_trace(result["input_results"], "input guardrails")
                    render_guardrail_trace(result["output_results"], "output guardrails")

        # ---- persist both turns (UNCHANGED history model) ----
        st.session_state.history.append(("user", user_msg))
        assistant_idx = len(st.session_state.history)
        st.session_state.history.append(("assistant", result["response"]))
        st.session_state.traces[assistant_idx] = result


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — ADMIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def metric_card(label: str, value: str, sub: str, accent: str, icon: str) -> str:
    return f"""
    <div class="glass" style="padding:1.15rem 1.3rem;height:100%;">
      <div style="display:flex;align-items:center;gap:0.5rem;color:#94a3b8;font-size:0.78rem;">
        <span style="font-size:1rem;">{icon}</span>{html.escape(label)}
      </div>
      <div style="font-size:2.05rem;font-weight:800;color:{accent};margin-top:0.35rem;
           line-height:1.1;">{value}</div>
      <div style="font-size:0.74rem;color:#6b7280;margin-top:0.15rem;">{html.escape(sub)}</div>
    </div>
    """


def normalize_guardrail_counts(by_guardrail: dict) -> dict:
    """Collapse 'output_filter/xyz' keys into their base guardrail bucket."""
    out: dict[str, int] = {}
    for key, count in by_guardrail.items():
        base = (key or "unknown").split("/")[0]
        out[base] = out.get(base, 0) + count
    return out


def render_dashboard_page() -> None:
    st.markdown(
        """
        <style>.main .block-container { max-width: 1180px; padding-top: 1rem; }</style>
        """,
        unsafe_allow_html=True,
    )

    stats = fetch_stats(DB_PATH)
    events = fetch_events(DB_PATH)   # newest-first, all rows
    total = stats["total"]

    # ---- time range subtitle ----
    if events:
        newest = datetime.fromtimestamp(events[0]["ts"])
        oldest = datetime.fromtimestamp(events[-1]["ts"])
        range_txt = f"{oldest:%Y-%m-%d %H:%M} → {newest:%Y-%m-%d %H:%M}"
    else:
        range_txt = "no events yet"

    st.markdown(
        f"<h2 style='margin-bottom:0.1rem;'>Monitoring Dashboard</h2>"
        f"<div style='color:#94a3b8;font-size:0.85rem;margin-bottom:1.2rem;'>"
        f"{total} total events · {html.escape(range_txt)}</div>",
        unsafe_allow_html=True,
    )

    # ---- Row 1: four metric cards ----
    blocked = stats["blocked"]
    allowed = stats["allowed"]
    avg_ms = stats.get("avg_latency_ms", 0.0) or 0.0
    blk_pct = (blocked / total * 100) if total else 0.0
    alw_pct = (allowed / total * 100) if total else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(metric_card("Total Messages", f"{total}", "all-time", BLUE, "💬"), unsafe_allow_html=True)
    m2.markdown(metric_card("Blocked", f"{blocked}", f"{blk_pct:.1f}% of traffic", ROSE, "🚫"), unsafe_allow_html=True)
    m3.markdown(metric_card("Allowed", f"{allowed}", f"{alw_pct:.1f}% of traffic", EMERALD, "✅"), unsafe_allow_html=True)
    m4.markdown(metric_card("Avg Latency", f"{avg_ms:.0f} ms", "per message", BLUE, "⚡"), unsafe_allow_html=True)

    st.markdown("<div style='height:1.3rem;'></div>", unsafe_allow_html=True)

    # ---- Row 2: donut (blocks by guardrail) + language bar ----
    norm_guardrail = normalize_guardrail_counts(stats["by_guardrail"])
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("<h4>Blocks by Guardrail</h4>", unsafe_allow_html=True)
        if norm_guardrail:
            labels = list(norm_guardrail.keys())
            values = list(norm_guardrail.values())
            colors = [GUARDRAIL_STYLE.get(l, (BLUE, l))[0] for l in labels]
            fig = go.Figure(go.Pie(
                labels=[GUARDRAIL_STYLE.get(l, (BLUE, l.upper()))[1] for l in labels],
                values=values, hole=0.62,
                marker=dict(colors=colors, line=dict(color="#0a0a0f", width=2)),
                textinfo="value", textfont=dict(color="#fff", size=13),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=True,
                              legend=dict(font=dict(size=11)))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("No blocks recorded yet — send a few unsafe prompts in Chat.")

    with c_right:
        st.markdown("<h4>Language Distribution</h4>", unsafe_allow_html=True)
        by_lang = stats["by_language"]
        if by_lang:
            order = [l for l in ("en", "hinglish", "hi") if l in by_lang]
            order += [l for l in by_lang if l not in order]
            counts = [by_lang[l] for l in order]
            names = [LANG_LABEL.get(l, l) for l in order]
            grad = ["#1e3a8a", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"]
            fig = go.Figure(go.Bar(
                x=counts, y=names, orientation="h",
                marker=dict(color=grad[:len(order)]),
                text=counts, textposition="outside", textfont=dict(color="#e5e7eb"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=300,
                              xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                              yaxis=dict(gridcolor="rgba(255,255,255,0.0)"))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("No messages yet.")

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

    # ---- Row 3: full-width blocks-by-category bar ----
    st.markdown("<h4>Blocks by Category</h4>", unsafe_allow_html=True)
    if norm_guardrail:
        labels = list(norm_guardrail.keys())
        values = list(norm_guardrail.values())
        colors = [GUARDRAIL_STYLE.get(l, (BLUE, l))[0] for l in labels]
        names = [GUARDRAIL_STYLE.get(l, (BLUE, l.upper()))[1] for l in labels]
        fig = go.Figure(go.Bar(
            x=names, y=values, marker=dict(color=colors),
            text=values, textposition="outside", textfont=dict(color="#e5e7eb"),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          xaxis=dict(gridcolor="rgba(255,255,255,0.0)"),
                          yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        st.info("No category firings yet.")

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

    # ---- Row 4: recent activity table (styled HTML) ----
    st.markdown("<h4>Recent Activity</h4>", unsafe_allow_html=True)
    recent = events[:30]
    if recent:
        body = ""
        for ev in recent:
            action = ev.get("final_action") or "?"
            if action == "blocked":
                a_badge = badge("Blocked", ROSE)
            else:
                a_badge = badge("Allowed", EMERALD)
            lang = LANG_LABEL.get(ev.get("language", ""), ev.get("language", "") or "?")
            blk = ev.get("blocked_by") or "—"
            preview = html.escape((ev.get("user_input") or "").replace("\n", " ")[:50])
            ms = ev.get("total_ms")
            ms_txt = f"{ms:.0f} ms" if isinstance(ms, (int, float)) else "—"
            body += (
                f"<tr style='border-top:1px solid rgba(255,255,255,0.05);'>"
                f"<td style='padding:8px 10px;color:#6b7280;'>{ev.get('id','')}</td>"
                f"<td style='padding:8px 10px;color:#cbd5e1;'>{html.escape(str(lang))}</td>"
                f"<td style='padding:8px 10px;'>{a_badge}</td>"
                f"<td style='padding:8px 10px;color:#f59e0b;'>{html.escape(str(blk))}</td>"
                f"<td style='padding:8px 10px;color:#9ca3af;'>{preview}</td>"
                f"<td style='padding:8px 10px;color:#94a3b8;text-align:right;'>{ms_txt}</td>"
                f"</tr>"
            )
        st.markdown(
            "<div class='glass' style='padding:0.4rem 0.6rem;overflow-x:auto;'>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.82rem;'>"
            "<thead><tr style='color:#6b7280;font-size:0.68rem;text-transform:uppercase;"
            "letter-spacing:0.05em;'>"
            "<th style='text-align:left;padding:6px 10px;'>ID</th>"
            "<th style='text-align:left;padding:6px 10px;'>Language</th>"
            "<th style='text-align:left;padding:6px 10px;'>Action</th>"
            "<th style='text-align:left;padding:6px 10px;'>Blocked By</th>"
            "<th style='text-align:left;padding:6px 10px;'>Input Preview</th>"
            "<th style='text-align:right;padding:6px 10px;'>Latency</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No events logged yet.")

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # ---- Download full log as CSV (reuses analyze_logs flatten logic) ----
    csv_bytes = build_log_csv(events)
    st.download_button(
        "⬇  Download Full Log (CSV)",
        data=csv_bytes,
        file_name=f"guardrail_log_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
    )


def build_log_csv(events: list[dict]) -> bytes:
    """In-memory CSV export mirroring analyze_logs.export_csv field layout."""
    fieldnames = [
        "id", "timestamp", "language", "user_input", "final_action",
        "blocked_by", "response", "total_ms",
        "input_guardrails", "output_guardrails",
    ]
    buf = io.StringIO()
    import csv as _csv
    writer = _csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for ev in reversed(events):   # chronological order, matches analyze_logs
        writer.writerow({
            "id": ev.get("id", ""),
            "timestamp": (
                datetime.fromtimestamp(ev["ts"]).isoformat() if ev.get("ts") else ""
            ),
            "language": ev.get("language") or "",
            "user_input": ev.get("user_input") or "",
            "final_action": ev.get("final_action") or "",
            "blocked_by": ev.get("blocked_by") or "",
            "response": ev.get("response") or "",
            "total_ms": ev.get("total_ms") or "",
            "input_guardrails": flatten_results(ev.get("input_results")),
            "output_guardrails": flatten_results(ev.get("output_results")),
        })
    return buf.getvalue().encode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════

if page.endswith("Chat"):
    render_chat_page()
else:
    render_dashboard_page()
