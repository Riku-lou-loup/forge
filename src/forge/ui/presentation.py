"""Shared presentation styles for the local investigation workspace."""

import streamlit as st

CSS = """
<style>
:root {--forge-brass:#ddba76;--forge-ink:#101816;--forge-muted:#b2c1b6;
--forge-line:#42564b;--forge-ease:cubic-bezier(.2,0,0,1);}
::selection {background:#ddba76;color:#101816;}
[data-testid="stMainBlockContainer"] {max-width:1320px;padding-top:2.7rem;padding-bottom:4rem;}
[data-testid="stSidebar"] {border-right:1px solid #34463d;}
[data-testid="stSidebar"] h1 {font-size:1.8rem;letter-spacing:.04em;}
h1 {font-size:2.5rem!important;letter-spacing:-.03em;font-weight:600!important;line-height:1.12!important;}
h2,h3 {letter-spacing:-.02em;}
p {line-height:1.6;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {color:var(--forge-muted)!important;opacity:1!important;font-size:.84rem;}
[data-testid="stMarkdownContainer"] p {max-width:78ch;}
[data-testid="stMetric"] {border-top:1px solid var(--forge-line);padding-top:.75rem;}
[data-testid="stMetricValue"] {font-size:1.65rem;letter-spacing:-.025em;font-variant-numeric:tabular-nums;}
[data-testid="stTabs"] [role="tablist"] {gap:1.8rem;border-bottom:1px solid var(--forge-line);}
[role="tab"] {min-height:48px;transition:color 100ms var(--forge-ease);}
button {transition:background-color 100ms var(--forge-ease),border-color 100ms var(--forge-ease),transform 140ms var(--forge-ease)!important;}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] * {color:#101816!important;}
button:active:not(:disabled) {transform:translateY(1px);}
button:focus-visible,[role="tab"]:focus-visible {outline:2px solid var(--forge-brass)!important;outline-offset:4px;}
input,textarea {caret-color:var(--forge-brass);}
a {text-underline-offset:4px;color:var(--forge-brass)!important;}
[data-testid="stExpander"] {border-color:var(--forge-line);border-radius:8px;}
.forge-rule {height:2px;background:#ddba76;margin:0 0 .8rem;transform-origin:left;animation:forge-arrive 600ms var(--forge-ease) both;}
.forge-note {color:var(--forge-muted);font-size:.85rem;line-height:1.6;border-top:1px solid var(--forge-line);padding-top:1rem;}
@keyframes forge-arrive {from{clip-path:inset(0 70% 0 0)}to{clip-path:inset(0)}}
@media(max-width:760px){[data-testid="stMainBlockContainer"]{padding:1.8rem 1rem 3rem;}h1{font-size:2rem!important;}[data-testid="stTabs"] [role="tablist"]{gap:1rem;}[data-testid="stMetricValue"]{font-size:1.3rem;}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important;}button:active{transform:none!important;}}
</style>
"""


def apply_style():
    st.html(CSS)


def chart_style(figure, height=350):
    figure.update_layout(
        height=height,
        paper_bgcolor="#101816",
        plot_bgcolor="#101816",
        font={"family": "sans-serif", "color": "#eff3eb", "size": 13},
        margin={"l": 16, "r": 24, "t": 38, "b": 32},
        legend={"orientation": "h", "y": 1.15},
        hoverlabel={"bgcolor": "#192521", "font_color": "#eff3eb"},
    )
    figure.update_xaxes(gridcolor="#293b31", zeroline=False)
    figure.update_yaxes(gridcolor="#293b31", zeroline=False)
    return figure
