"""Global layout helpers: CSS injection, sidebar branding, page headers."""

from pathlib import Path

import streamlit as st

from utils.paths import ASSETS_DIR


# ── CSS ─────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    """Read assets/styles.css and inject it into the Streamlit page."""
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found: {css_path}")


# ── Sidebar ──────────────────────────────────────────────────────────────────

def sidebar_branding(
    winner_model: str = "—",
    winner_f1: str = "—",
    pipeline_ok: bool = True,
) -> None:
    """Render the custom sidebar branding block (title, subtitle, status).

    Uses st.sidebar.markdown() so that CSS classes from styles.css apply
    (st.sidebar.html() renders in an isolated iframe where global CSS does not reach).
    """
    status_color = "var(--ok)" if pipeline_ok else "var(--err)"
    status_text  = "&#x25CF; ok" if pipeline_ok else "&#x25CF; error"

    st.sidebar.markdown(
        '<div class="sb-brand">'
        '<div class="sb-title">ECG Arrhythmia ML</div>'
        '<div class="sb-sub">Clasificación de ritmos intraoperatorios</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        f'<div class="sb-status">'
        f'<div style="margin-bottom:8px">'
        f'<span class="sb-stamp">DEMO ACADÉMICA</span>'
        f'</div>'
        f'<div class="sb-status-row">'
        f'<span class="sb-status-key">pipeline</span>'
        f'<span style="color:{status_color}">{status_text}</span>'
        f'</div>'
        f'<div class="sb-status-row">'
        f'<span class="sb-status-key">best model</span>'
        f'<span class="sb-status-val">{winner_model}</span>'
        f'</div>'
        f'<div class="sb-status-row">'
        f'<span class="sb-status-key">F1-macro</span>'
        f'<span class="sb-status-val">{winner_f1}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Page Header ──────────────────────────────────────────────────────────────

def page_header(
    title: str,
    subtitle: str = "",
    badge_html: str = "",
) -> None:
    """Render a consistent page header with title, lead text and optional badge."""
    badge_block = (
        f'<div style="margin-bottom:8px">{badge_html}</div>' if badge_html else ""
    )
    lead_p = f'<p class="lead">{subtitle}</p>' if subtitle else ""

    st.html(
        f'<div class="page-head">'
        f'{badge_block}'
        f'<h2>{title}</h2>'
        f'{lead_p}'
        f'</div>'
    )


# ── Placeholder page ─────────────────────────────────────────────────────────

def placeholder_page(
    title: str,
    description: str,
    files_needed: list[str] | None = None,
    note: str = "",
) -> None:
    """Render a full placeholder page for sections not yet implemented."""
    page_header(title, description)

    files_html = ""
    if files_needed:
        items = "".join(
            f'<li style="font-family:var(--mono);font-size:12px;color:var(--fg-2);margin:4px 0">{f}</li>'
            for f in files_needed
        )
        files_html = (
            '<div style="margin-top:14px;font-size:12.5px;color:var(--fg-3)">'
            '<div style="font-family:var(--mono);text-transform:uppercase;'
            'letter-spacing:.08em;font-size:10.5px;margin-bottom:6px">'
            "Archivos que usará esta sección:"
            "</div>"
            f'<ul style="margin:0;padding-left:18px">{items}</ul>'
            "</div>"
        )

    note_html = (
        f'<p style="margin-top:12px;font-size:12px;color:var(--fg-4)">{note}</p>'
        if note else ""
    )

    st.html(
        f'<div class="placeholder-block">'
        f'<div class="ph-mono">sección en construcción</div>'
        f'<div class="ph-title">{title}</div>'
        f'<div class="ph-desc">{description}</div>'
        f'{files_html}'
        f'{note_html}'
        f'</div>'
    )
