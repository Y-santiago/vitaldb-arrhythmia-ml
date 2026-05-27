"""HTML badge helpers — return strings for use with st.markdown(unsafe_allow_html=True)."""

_KINDS = {"ok", "warn", "err", "info", "muted", "winner"}


def badge(text: str, kind: str = "muted") -> str:
    """Return an inline HTML badge span."""
    kind = kind if kind in _KINDS else "muted"
    return f'<span class="badge badge-{kind}">{text}</span>'


def badge_row(*badges: str) -> str:
    """Wrap multiple badge strings in a flex row div."""
    return f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin:6px 0">{"".join(badges)}</div>'


def status_badge(text: str, status: str = "ok") -> str:
    """Convenience wrapper — maps status strings to badge kinds."""
    mapping = {"ok": "ok", "warning": "warn", "error": "err", "info": "info"}
    kind = mapping.get(status, "muted")
    return badge(text, kind)


def pill(text: str, kind: str = "muted") -> str:
    """Round pill — same colours as badge but fully rounded."""
    kind = kind if kind in _KINDS else "muted"
    extra = "border-radius:999px;"
    return f'<span class="badge badge-{kind}" style="{extra}">{text}</span>'


def class_status_badge(f1: float, support: int, low_support_threshold: int = 500) -> str:
    """Return a badge indicating class performance tier.

    Tiers: winner (f1≥0.70), info (f1≥0.30), warn (f1<0.30 and support≥threshold),
           err (f1<0.30 and support<threshold — low data class).
    """
    if f1 >= 0.70:
        return badge(f"F1 {f1:.3f}", "ok")
    if f1 >= 0.30:
        return badge(f"F1 {f1:.3f}", "info")
    if support < low_support_threshold:
        return badge(f"F1 {f1:.3f} · bajo soporte", "err")
    return badge(f"F1 {f1:.3f}", "warn")
