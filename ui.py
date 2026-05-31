import streamlit as st
from typing import Dict, Iterable, Optional, Union


def inject_global_style(theme: str = "light") -> None:
    body_bg = "#f8fafc"
    surface_bg = "#ffffff"
    text_primary = "#0f172a"
    text_secondary = "#475569"
    border_color = "#e2e8f0"
    accent = "#2563eb"
    card_shadow = "rgba(15, 23, 42, 0.08)"

    st.markdown(
        f"""
        <style>
        .app-content {{
            color: {text_primary};
        }}
        .app-header {{
            padding: 18px 0 12px 0;
            margin-bottom: 24px;
        }}
        .app-title {{
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.1rem;
            color: {text_primary};
        }}
        .app-subtitle {{
            color: {text_secondary};
            font-size: 1rem;
            margin-top: 0.35rem;
            line-height: 1.6;
        }}
        .card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }}
        .metric-card, .overview-card {{
            background: {surface_bg};
            border: 1px solid {border_color};
            border-radius: 24px;
            padding: 22px;
            box-shadow: 0 18px 40px {card_shadow};
            min-height: 140px;
        }}
        .metric-card-title {{
            font-size: 0.95rem;
            color: {text_secondary};
            margin-bottom: 0.3rem;
        }}
        .metric-card-value {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.65rem;
            color: {text_primary};
        }}
        .metric-card-delta {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            color: {accent};
            font-weight: 600;
        }}
        .metric-card-delta span {{
            font-size: 1.05rem;
        }}
        .section-header {{
            margin-top: 24px;
            margin-bottom: 14px;
        }}
        .section-title {{
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            color: {text_primary};
        }}
        .section-description {{
            color: {text_secondary};
            font-size: 0.95rem;
            margin: 0;
        }}
        .app-note {{
            color: {text_secondary};
            margin: 16px 0 0 0;
            font-size: 0.95rem;
        }}
        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.12);
            color: {accent};
            font-size: 0.9rem;
            font-weight: 600;
            border: 1px solid rgba(56, 189, 248, 0.18);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-title">{title}</div>
            <div class="app-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(cards: Iterable[Dict[str, Union[str, None]]]) -> None:
    html = ''
    for card in cards:
        delta = card.get('delta')
        delta_html = f"<div class='metric-card-delta'>{delta}</div>" if delta else ''
        caption = card.get('caption', '') or ''
        html += (
            f"<div class='metric-card'>"
            f"<div class='metric-card-title'>{card['title']}</div>"
            f"<div class='metric-card-value'>{card['value']}</div>"
            f"{delta_html}"
            f"<div class='app-note'>{caption}</div>"
            f"</div>"
        )

    st.markdown(f"<div class='card-grid'>{html}</div>", unsafe_allow_html=True)


def render_section_header(title: str, description: Optional[str] = None) -> None:
    content = f"<div class='section-header'><div class='section-title'>{title}</div>"
    if description:
        content += f"<div class='section-description'>{description}</div>"
    content += "</div>"
    st.markdown(content, unsafe_allow_html=True)


def render_table_section(title: str, df, details: Optional[str] = None) -> None:
    render_section_header(title, details)
    st.dataframe(df, width="100%")
