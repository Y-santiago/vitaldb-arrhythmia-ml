"""Chart components — Plotly-based with dark ECG theme."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Shared Plotly layout defaults ────────────────────────────────────────────

_TRANSPARENT = "rgba(0,0,0,0)"
_BG_PLOT     = "#121a2b"
_GRID_COLOR  = "rgba(255,255,255,0.05)"
_AXIS_COLOR  = "#3d4b6b"
_TEXT_COLOR  = "#8a98b5"
_FONT_MONO   = "IBM Plex Mono, monospace"

DARK_LAYOUT = dict(
    plot_bgcolor=_BG_PLOT,
    paper_bgcolor=_TRANSPARENT,
    font=dict(family=_FONT_MONO, color=_TEXT_COLOR, size=11),
    margin=dict(l=10, r=10, t=36, b=10),
    showlegend=False,
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        color=_AXIS_COLOR,
        tickfont=dict(family=_FONT_MONO, size=10),
    ),
    yaxis=dict(
        gridcolor=_GRID_COLOR,
        zeroline=False,
        color=_AXIS_COLOR,
        tickfont=dict(family=_FONT_MONO, size=10),
    ),
)


def apply_dark_layout(fig: go.Figure, **overrides) -> go.Figure:
    """Apply DARK_LAYOUT to fig, with overrides taking precedence.

    Prevents duplicate-kwarg TypeError when callers need to override
    keys already present in DARK_LAYOUT (margin, xaxis, yaxis, showlegend…).
    """
    layout = dict(DARK_LAYOUT)
    layout.update(overrides)
    fig.update_layout(**layout)
    return fig


# ── ECG mini strip (synthetic) ────────────────────────────────────────────────

def _ecg_beat(t: np.ndarray, t0: float) -> np.ndarray:
    """Single synthetic ECG beat: P + QRS + T waves."""
    dt = t - t0
    p  =  0.15 * np.exp(-((dt - 0.10) ** 2) / 0.0012)
    q  = -0.10 * np.exp(-((dt - 0.18) ** 2) / 0.0002)
    r  =  1.00 * np.exp(-((dt - 0.20) ** 2) / 0.00015)
    s  = -0.18 * np.exp(-((dt - 0.22) ** 2) / 0.0002)
    tw =  0.28 * np.exp(-((dt - 0.36) ** 2) / 0.0032)
    return p + q + r + s + tw


def mini_ecg_placeholder(height: int = 140, n_beats: int = 6) -> go.Figure:
    """Generate a synthetic ECG strip using Plotly. No real data required."""
    rng = np.random.default_rng(42)
    t   = np.linspace(0, 1, 800)
    sig = np.zeros(len(t))

    positions = np.linspace(0.08, 0.92, n_beats)
    for t0 in positions:
        sig += _ecg_beat(t, t0)
    sig += rng.normal(0, 0.018, len(t))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=sig,
            mode="lines",
            line=dict(color="#2dd4bf", width=1.6, shape="spline", smoothing=0.3),
            fill="tozeroy",
            fillcolor="rgba(45,212,191,0.05)",
        )
    )
    apply_dark_layout(
        fig,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(
            showgrid=True,
            gridcolor=_GRID_COLOR,
            showticklabels=False,
            zeroline=False,
        ),
    )
    return fig


# ── Empty placeholder chart ────────────────────────────────────────────────────

def empty_chart_placeholder(title: str = "", description: str = "") -> None:
    """Render an HTML placeholder where a chart will eventually go."""
    st.markdown(
        f"""<div class="placeholder-block" style="min-height:220px">
              <div class="ph-mono">gráfico pendiente</div>
              <div class="ph-title">{title}</div>
              <div class="ph-desc">{description}</div>
            </div>""",
        unsafe_allow_html=True,
    )


# ── Horizontal bar chart ───────────────────────────────────────────────────────

def bar_chart_h(
    labels: list[str],
    values: list[float],
    title: str = "",
    color: str = "#4a8cff",
    accent_top: bool = True,
    value_fmt: str = ".3f",
    height: int = 280,
    log_x: bool = False,
) -> go.Figure:
    """Horizontal bar chart sorted descending, accent colour on top bar.

    Args:
        log_x: if True, use logarithmic x-axis (useful for support charts).
    """
    pairs = sorted(zip(labels, values), key=lambda x: x[1])
    lbls, vals = zip(*pairs) if pairs else ([], [])

    colors = [
        "#2dd4bf" if (accent_top and i == len(vals) - 1) else color
        for i in range(len(vals))
    ]

    text_pos = "auto" if log_x else "outside"

    fig = go.Figure(
        go.Bar(
            x=list(vals),
            y=list(lbls),
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:{value_fmt}}" for v in vals],
            textposition=text_pos,
            textfont=dict(family=_FONT_MONO, size=10, color=_TEXT_COLOR),
            cliponaxis=False,
        )
    )
    x_kwargs = dict(
        showgrid=True,
        gridcolor=_GRID_COLOR,
        zeroline=False,
        showticklabels=log_x,
    )
    if log_x:
        x_kwargs["type"] = "log"

    apply_dark_layout(
        fig,
        title=dict(text=title, font=dict(size=12, color="#c8d2e5")) if title else {},
        height=height,
        bargap=0.3,
        xaxis=x_kwargs,
        yaxis=dict(showgrid=False, zeroline=False, autorange=True),
    )
    return fig


# ── Per-class F1 horizontal bar ────────────────────────────────────────────────

def class_f1_bar(
    labels: list[str],
    f1_values: list[float],
    thresholds: tuple[float, float] = (0.70, 0.30),
    height: int = 340,
) -> go.Figure:
    """Horizontal bar chart for per-class F1, colour-coded by performance tier.

    Colours:
        teal  -> F1 >= high_thr  (good)
        blue  -> low_thr <= F1 < high_thr  (medium)
        red   -> F1 < low_thr  (difficult)
    """
    high_thr, low_thr = thresholds
    pairs = sorted(zip(labels, f1_values), key=lambda x: x[1])
    sorted_labels, sorted_f1 = zip(*pairs) if pairs else ([], [])

    def _color(v: float) -> str:
        if v >= high_thr:
            return "#2dd4bf"
        if v >= low_thr:
            return "#4a8cff"
        return "#f87171"

    fig = go.Figure(
        go.Bar(
            x=list(sorted_f1),
            y=list(sorted_labels),
            orientation="h",
            marker=dict(color=[_color(v) for v in sorted_f1], line=dict(width=0)),
            text=[f"{v:.3f}" for v in sorted_f1],
            textposition="outside",
            textfont=dict(family=_FONT_MONO, size=10, color=_TEXT_COLOR),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>F1-score: %{x:.3f}<extra></extra>",
        )
    )

    fig.add_vline(
        x=high_thr,
        line=dict(color="rgba(45,212,191,0.35)", dash="dot", width=1.5),
        annotation_text=f"F1={high_thr}",
        annotation_font=dict(color="rgba(45,212,191,0.65)", size=9, family=_FONT_MONO),
        annotation_position="top right",
    )
    fig.add_vline(
        x=low_thr,
        line=dict(color="rgba(248,113,113,0.35)", dash="dot", width=1.5),
        annotation_text=f"F1={low_thr}",
        annotation_font=dict(color="rgba(248,113,113,0.65)", size=9, family=_FONT_MONO),
        annotation_position="top right",
    )

    apply_dark_layout(
        fig,
        height=height,
        bargap=0.22,
        xaxis=dict(
            range=[0, 1.08],
            showgrid=True,
            gridcolor=_GRID_COLOR,
            zeroline=False,
            tickvals=[0, 0.25, 0.50, 0.75, 1.0],
            showticklabels=True,
            tickfont=dict(family=_FONT_MONO, size=9),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            autorange=True,
            tickfont=dict(size=11),
        ),
        margin=dict(l=10, r=30, t=10, b=10),
    )
    return fig


# ── Support vs F1 scatter ─────────────────────────────────────────────────────

def support_vs_f1_scatter(
    labels: list[str],
    support_values: list[float],
    f1_values: list[float],
    thresholds: tuple[float, float] = (0.70, 0.30),
    height: int = 380,
) -> go.Figure:
    """Scatter plot: support (X, log scale) vs F1-score (Y).

    Each point is a class. Colour encodes performance tier.
    """
    high_thr, low_thr = thresholds

    def _color(v: float) -> str:
        if v >= high_thr:
            return "#2dd4bf"
        if v >= low_thr:
            return "#4a8cff"
        return "#f87171"

    colors = [_color(v) for v in f1_values]

    fig = go.Figure(
        go.Scatter(
            x=support_values,
            y=f1_values,
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(family=_FONT_MONO, size=9, color=_TEXT_COLOR),
            marker=dict(
                size=14,
                color=colors,
                line=dict(width=1, color="rgba(255,255,255,0.12)"),
                opacity=0.9,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Support: %{x:,.0f}<br>"
                "F1-score: %{y:.3f}<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=low_thr,
        line=dict(color="rgba(248,113,113,0.35)", dash="dash", width=1.5),
        annotation_text=f"umbral dificil (F1={low_thr})",
        annotation_font=dict(color="rgba(248,113,113,0.6)", size=9, family=_FONT_MONO),
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=high_thr,
        line=dict(color="rgba(45,212,191,0.25)", dash="dash", width=1.5),
        annotation_text=f"umbral buen desempeno (F1={high_thr})",
        annotation_font=dict(color="rgba(45,212,191,0.5)", size=9, family=_FONT_MONO),
        annotation_position="top right",
    )

    apply_dark_layout(
        fig,
        height=height,
        xaxis=dict(
            type="log",
            showgrid=True,
            gridcolor=_GRID_COLOR,
            zeroline=False,
            title=dict(text="Support (ventanas · escala log)", font=dict(size=11)),
            tickfont=dict(family=_FONT_MONO, size=9),
        ),
        yaxis=dict(
            range=[-0.06, 1.06],
            showgrid=True,
            gridcolor=_GRID_COLOR,
            zeroline=False,
            title=dict(text="F1-score", font=dict(size=11)),
            tickfont=dict(family=_FONT_MONO, size=9),
        ),
        margin=dict(l=55, r=20, t=20, b=55),
    )
    return fig


# ── Grouped vertical bar chart ────────────────────────────────────────────────

def bar_chart_v_grouped(
    df,
    x_col: str,
    y_cols: list[str],
    colors: list[str] | None = None,
    title: str = "",
    height: int = 280,
    y_range: list | None = None,
) -> go.Figure:
    """Grouped vertical bar chart from a DataFrame."""
    default_colors = ["#4a8cff", "#2dd4bf", "#fbbf24", "#f87171", "#a78bfa"]
    colors = colors or default_colors[: len(y_cols)]

    fig = go.Figure()
    for col, color in zip(y_cols, colors):
        fig.add_trace(
            go.Bar(
                name=col,
                x=df[x_col],
                y=df[col],
                marker_color=color,
                marker_line_width=0,
            )
        )

    apply_dark_layout(
        fig,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            font=dict(family=_FONT_MONO, size=10),
        ),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        title=dict(text=title, font=dict(size=12, color="#c8d2e5")) if title else {},
        height=height,
        yaxis=dict(range=y_range, showgrid=True, gridcolor=_GRID_COLOR, zeroline=False),
    )
    return fig


# ── Model metrics grouped bar ─────────────────────────────────────────────────

def model_metrics_bar(
    labels: list[str],
    metrics: dict[str, list[float]],
    winner_label: str = "",
    height: int = 300,
) -> go.Figure:
    """Grouped vertical bar chart for model comparison with winner highlight."""
    _colors = ["#2dd4bf", "#4a8cff", "#fbbf24", "#a78bfa", "#f87171"]
    fig = go.Figure()

    for (metric_name, values), color in zip(metrics.items(), _colors):
        fig.add_trace(
            go.Bar(
                name=metric_name,
                x=labels,
                y=values,
                marker_color=color,
                marker_line_width=0,
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
                textfont=dict(family=_FONT_MONO, size=9, color=_TEXT_COLOR),
                cliponaxis=False,
            )
        )

    shapes = []
    if winner_label and winner_label in labels:
        wi = labels.index(winner_label)
        shapes = [
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=wi - 0.48,
                x1=wi + 0.48,
                y0=0,
                y1=1,
                fillcolor="rgba(45,212,191,0.06)",
                line=dict(color="rgba(45,212,191,0.18)", width=1),
                layer="below",
            )
        ]

    apply_dark_layout(
        fig,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            font=dict(family=_FONT_MONO, size=10),
        ),
        barmode="group",
        bargap=0.22,
        bargroupgap=0.08,
        height=height,
        shapes=shapes,
        yaxis=dict(range=[0, 1.05], showgrid=True, gridcolor=_GRID_COLOR, zeroline=False),
        margin=dict(l=10, r=10, t=44, b=10),
    )
    return fig


# ── Fit time horizontal bar ───────────────────────────────────────────────────

def fit_time_bar(
    labels: list[str],
    times: list[float],
    winner_label: str = "",
    height: int = 240,
) -> go.Figure:
    """Horizontal bar chart showing model training time sorted ascending."""
    pairs = sorted(zip(labels, times), key=lambda x: x[1])
    sorted_labels, sorted_times = zip(*pairs) if pairs else ([], [])

    bar_colors = [
        "#2dd4bf" if lbl == winner_label else "#4a8cff"
        for lbl in sorted_labels
    ]

    def _fmt_time(t: float) -> str:
        if t >= 3600:
            return f"{t / 3600:.1f} h"
        if t >= 60:
            return f"{t / 60:.0f} min"
        return f"{t:.0f} s"

    fig = go.Figure(
        go.Bar(
            x=list(sorted_times),
            y=list(sorted_labels),
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[_fmt_time(t) for t in sorted_times],
            textposition="outside",
            textfont=dict(family=_FONT_MONO, size=10, color=_TEXT_COLOR),
            cliponaxis=False,
        )
    )
    apply_dark_layout(
        fig,
        height=height,
        bargap=0.32,
        xaxis=dict(
            showgrid=True,
            gridcolor=_GRID_COLOR,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(showgrid=False, zeroline=False),
        margin=dict(l=10, r=60, t=10, b=10),
    )
    return fig
