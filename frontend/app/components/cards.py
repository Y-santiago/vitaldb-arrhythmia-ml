"""Card and metric components for Streamlit pages."""

import streamlit as st


def card_header(title: str, subtitle: str = "", right_html: str = "") -> None:
    """Render a styled card header (title + optional subtitle + right slot)."""
    sub_html = f'<span class="card-sub">· {subtitle}</span>' if subtitle else ""
    st.html(
        f'<div class="card-header">'
        f'<div><span class="card-title">{title}</span>{sub_html}</div>'
        f'<div>{right_html}</div>'
        f'</div>'
    )


def metric_card(
    label: str,
    value: str,
    helper: str = "",
    accent: str = "blue",
    helper_kind: str = "ok",
) -> None:
    """Render a custom HTML metric card with accent border."""
    helper_html = (
        f'<div class="metric-helper {helper_kind}">{helper}</div>' if helper else ""
    )
    st.html(
        f'<div class="metric-card accent-{accent}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{helper_html}'
        f'</div>'
    )


def callout(kind: str, title: str, body: str) -> None:
    """Render a styled callout / alert block.
    kind: 'info' | 'warn' | 'err' | 'ok'
    """
    icons = {"info": "ℹ", "warn": "⚠", "err": "✕", "ok": "✓"}
    icon = icons.get(kind, "ℹ")
    st.html(
        f'<div class="callout-block callout-{kind}">'
        f'<span class="callout-title">{icon}&nbsp;{title}</span>'
        f'<div>{body}</div>'
        f'</div>'
    )


def kv_table(rows: list[tuple]) -> None:
    """Render a key-value definition table.
    rows: list of (key, value) tuples.
    """
    items = "".join(
        f'<div class="kv-key">{k}</div><div class="kv-val">{v}</div>'
        for k, v in rows
    )
    st.html(f'<div class="kv-table">{items}</div>')


def section_title(text: str) -> None:
    """Render a monospace section divider / label."""
    st.html(f'<div class="section-title">{text}</div>')
