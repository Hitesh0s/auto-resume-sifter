"""Shared UI helpers: CSS blocks, navbar HTML, and formatting functions."""

from __future__ import annotations

import pandas as pd

# ── Global CSS ────────────────────────────────────────────────────────────────
CSS_BLOCK = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Remove ALL border-radius; badges get 2px only */
*, *::before, *::after { border-radius: 0 !important; }
.badge { border-radius: 2px !important; }

/* App background */
.stApp, [data-testid="stAppViewContainer"] { background: #f2f5f9 !important; }

/* Hide sidebar chrome */
[data-testid="stSidebar"]         { display: none !important; }
[data-testid="collapsedControl"]  { display: none !important; }

/* Content area */
.main .block-container,
[data-testid="stMainBlockContainer"] {
    padding-top: 0.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1300px !important;
}

/* ── Top navigation bar ── */
.ars-nav-divider {
    border: none; border-top: 4px solid #0c1a2e; margin: 0 0 1.5rem;
}
.ars-brand-text {
    font-size: 1.05rem; font-weight: 800; color: #0c1a2e;
    letter-spacing: -0.03em; padding-top: 7px; line-height: 1.1;
}
.ars-brand-text em { color: #C8970A; font-style: normal; }
.ars-user-info { font-size: 0.72rem; color: #5e738a; padding-top: 10px; white-space: nowrap; }

/* Style page links as nav tabs */
[data-testid="stPageLink"] { padding: 0 !important; margin: 0 !important; }
[data-testid="stPageLink"] a {
    color: #475569 !important;
    text-decoration: none !important;
    font-weight: 600 !important; font-size: 0.82rem !important;
    padding: 6px 12px !important;
    display: flex !important; align-items: center !important;
    border-bottom: 3px solid transparent !important;
    height: 40px !important; white-space: nowrap !important;
    letter-spacing: 0.01em !important;
    transition: color 0.12s, border-color 0.12s !important;
}
[data-testid="stPageLink"] a:hover {
    color: #1E3A5F !important;
    border-bottom-color: #C8970A !important;
    text-decoration: none !important;
}
[data-testid="stPageLink"] a[aria-current="page"] {
    color: #0c1a2e !important; font-weight: 700 !important;
    border-bottom-color: #C8970A !important;
}

/* ── Markdown section headers (###) ── */
h3 {
    font-size: 1rem !important; font-weight: 700 !important;
    color: #0c1825 !important; letter-spacing: -0.01em !important;
    margin: 0.5rem 0 0.6rem !important;
    padding-left: 0.65rem !important;
    border-left: 3px solid #C8970A !important;
}
h2 {
    font-size: 1.15rem !important; font-weight: 800 !important;
    color: #0c1825 !important; letter-spacing: -0.02em !important;
    margin: 0.75rem 0 0.5rem !important;
}

/* ── Section typography ── */
.section-title {
    font-size: 1.25rem; font-weight: 800; color: #0c1825;
    letter-spacing: -0.02em; margin: 0 0 0.15rem;
    border-left: 3px solid #C8970A; padding-left: 0.75rem;
}
.section-sub {
    font-size: 0.875rem; color: #5e738a;
    margin: 0 0 1.25rem; padding-left: 0.75rem;
}
.section-heading {
    font-size: 0.82rem; font-weight: 700; color: #1E3A5F;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 1.5rem 0 0.75rem; padding-bottom: 0.4rem;
    border-bottom: 2px solid #C8970A; display: inline-block;
}
.page-subtitle { font-size: 0.875rem; color: #5e738a; margin-bottom: 1.25rem; }

/* Divider */
.divider { border: none; border-top: 1px solid #dde4ee; margin: 1.75rem 0; }

/* ── Stat cards ── */
.stat-card {
    background: #fff; border: 1px solid #dde4ee;
    border-top: 3px solid #1E3A5F;
    padding: 1.1rem 1.25rem; text-align: center;
    transition: border-top-color 0.15s;
}
.stat-card:hover { border-top-color: #C8970A; }
.stat-card .stat-label {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #8a9ab0; margin-bottom: 0.4rem;
}
.stat-card .stat-value {
    font-size: 1.85rem; font-weight: 800; color: #1E3A5F;
    line-height: 1; letter-spacing: -0.03em;
}

/* ── Tier badges ── */
.badge {
    display: inline-block; padding: 2px 8px;
    font-size: 0.68rem; font-weight: 700;
    white-space: nowrap; letter-spacing: 0.04em; text-transform: uppercase;
}
.badge-strong     { background: #e8f5ec; color: #155724; border: 1px solid #b4dfc0; }
.badge-partial    { background: #fff8e6; color: #7d5a00; border: 1px solid #f0d080; }
.badge-unsuitable { background: #fce9e9; color: #8b1a1a; border: 1px solid #e8a8a8; }

/* ── Skill tags ── */
.skill-tag {
    display: inline-block; background: #eef2f9; color: #1E3A5F;
    border: 1px solid #c8d4e8; padding: 1px 7px;
    font-size: 0.7rem; font-weight: 500; margin: 2px 2px 2px 0;
}
.skill-tag-green {
    display: inline-block; background: #e8f5ec; color: #155724;
    border: 1px solid #b4dfc0; padding: 1px 7px;
    font-size: 0.7rem; font-weight: 500; margin: 2px 2px 2px 0;
}
.skill-tag-orange {
    display: inline-block; background: #fff8e6; color: #7d5a00;
    border: 1px solid #f0d080; padding: 1px 7px;
    font-size: 0.7rem; font-weight: 500; margin: 2px 2px 2px 0;
}

/* ── Results table ── */
.results-table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
.results-table th {
    background: #0c1825; color: #9ab0c8; font-weight: 700;
    font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.65rem 0.875rem; border-bottom: 2px solid #C8970A;
    text-align: left; white-space: nowrap;
}
.results-table td {
    padding: 0.6rem 0.875rem; border-bottom: 1px solid #eef2f9;
    color: #1a2940; vertical-align: top; background: #fff;
}
.results-table tbody tr:nth-child(even) td { background: #f8fafc; }
.results-table tbody tr:hover td         { background: #eef2f9 !important; }
.results-table tr:last-child td          { border-bottom: none; }
.score-cell { font-weight: 700; font-variant-numeric: tabular-nums; color: #1E3A5F; }
.top-score  { color: #C8970A !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1px solid #dde4ee; background: #fff; padding: 0.5rem;
}

/* ── Buttons ── */
.stButton > button {
    background: #1E3A5F; color: #fff; border: none;
    font-weight: 600; font-size: 0.875rem; padding: 0.5rem 1.5rem;
    letter-spacing: 0.01em; transition: background 0.15s;
}
.stButton > button:hover { background: #2a5080 !important; color: #fff; }
.stButton > button[kind="primary"] { background: #1E3A5F; box-shadow: none; }
.stButton > button[kind="primary"]:hover { background: #2a5080 !important; box-shadow: none !important; }

/* ── Inputs ── */
.stTextInput > div > div, .stTextArea > div > div {
    border: 1px solid #dde4ee; background: #fff;
}
.stTextInput > div > div:focus-within,
.stTextArea > div > div:focus-within {
    border-color: #1E3A5F !important;
    box-shadow: 0 0 0 1px rgba(30,58,95,0.18) !important;
}

/* ── Callout boxes ── */
.info-box {
    background: #eef2f9; border-left: 4px solid #1E3A5F;
    padding: 0.75rem 1rem; font-size: 0.875rem; color: #1E3A5F;
}
.success-box {
    background: #e8f5ec; border-left: 4px solid #155724;
    padding: 0.75rem 1rem; font-size: 0.875rem; color: #155724;
}
.warn-box {
    background: #fff8e6; border-left: 4px solid #C8970A;
    padding: 0.75rem 1rem; font-size: 0.875rem; color: #7d5a00;
}
.error-box {
    background: #fce9e9; border-left: 4px solid #8b1a1a;
    padding: 0.75rem 1rem; font-size: 0.875rem; color: #8b1a1a;
}
.note-box {
    background: #f8fafc; border: 1px solid #dde4ee;
    padding: 0.875rem 1rem; font-size: 0.8rem; color: #5e738a; line-height: 1.6;
}
.bias-clean {
    background: #e8f5ec; border-left: 4px solid #155724;
    padding: 0.75rem 1rem; font-size: 0.875rem; color: #155724;
}
.bias-warn {
    background: #fff8e6; border-left: 4px solid #d97706;
    padding: 0.75rem 1rem; font-size: 0.875rem; color: #7d5a00;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 2px solid #dde4ee; }
[data-testid="stTabs"] [role="tab"] {
    font-weight: 600; font-size: 0.82rem; color: #5e738a;
    padding: 0.55rem 1.2rem; border-bottom: 2px solid transparent;
    margin-bottom: -2px; letter-spacing: 0.01em;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #1E3A5F !important; border-bottom-color: #C8970A !important;
}

/* ── Expander ── */
[data-testid="stExpander"] { border: 1px solid #dde4ee; background: #fff; }
[data-testid="stExpander"] summary {
    font-weight: 600; font-size: 0.875rem; color: #1E3A5F;
    padding: 0.7rem 1rem; background: #f8fafc; border-bottom: 1px solid #dde4ee;
}
[data-testid="stExpander"] summary:hover { background: #eef2f9; }

/* ── Plotly ── */
.js-plotly-plot { border: 1px solid #dde4ee; background: #fff; }

/* ── Progress bar ── */
[data-testid="stProgressBarTrack"] > div { background: #1E3A5F !important; }
[data-testid="stProgress"] p,
[data-testid="stProgress"] span,
[data-testid="stProgress"] label { color: #1E3A5F !important; font-weight: 500 !important; }

/* ── Demo banner ── */
.demo-banner {
    background: linear-gradient(135deg, #0c1a2e 0%, #1a3558 100%);
    border: 1px solid rgba(200,151,10,0.25);
    border-left: 4px solid #C8970A;
    padding: 0.85rem 1.25rem;
    display: flex; align-items: center; min-height: 60px;
}

/* ── Top-scorer card ── */
.top-scorer-card {
    margin-top: 1.25rem; border: 2px solid #C8970A;
    padding: 1.25rem 1.5rem;
    background: linear-gradient(135deg, #fffdf0 0%, #fff 100%);
}

/* ── Login form ── */
[data-testid="stForm"] {
    border: 1px solid #dde4ee !important; border-top: 3px solid #C8970A !important;
    padding: 2rem 2.25rem; max-width: 420px; margin: 0 auto;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label { font-size: 0.84rem; color: #2a3f5c; }
[data-testid="stCheckbox"] label:hover { color: #1E3A5F; }

/* ── Scrollbar ── */
::-webkit-scrollbar             { width: 5px; height: 5px; }
::-webkit-scrollbar-track       { background: #f2f5f9; }
::-webkit-scrollbar-thumb       { background: #c8d4e8; }
::-webkit-scrollbar-thumb:hover { background: #8a9ab0; }

:focus-visible { outline: 2px solid #1E3A5F; outline-offset: 2px; }
* { box-sizing: border-box; }
</style>"""

# ── Login-page dark background ─────────────────────────────────────────────────
LOGIN_CSS = """<style>
.stApp, [data-testid="stAppViewContainer"], body, html {
    background: linear-gradient(140deg, #0a1628 0%, #1a3558 100%) !important;
}
[data-testid="stMain"] { background: transparent !important; }
[data-testid="stForm"] {
    background: #fff !important;
    border: 1px solid #2a4a70 !important;
    border-top: 3px solid #C8970A !important;
    box-shadow: 0 8px 48px rgba(0,0,0,0.55) !important;
    max-width: 420px !important;
    margin: 0 auto !important;
    padding: 2rem 2.25rem !important;
}
[data-testid="stForm"] label,
[data-testid="stForm"] p { color: #1a2940 !important; }
</style>"""

# ── Login brand panel (injected above the login form) ─────────────────────────
BRANDING_HTML = """
<div style="padding:1.5rem 1rem 0.75rem;text-align:center;max-width:700px;margin:0 auto;">
  <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;
              color:#C8970A;margin-bottom:0.5rem;">HR Intelligence Platform</div>
  <div style="font-size:2.2rem;font-weight:800;color:#fff;line-height:1.1;
              margin-bottom:0.4rem;letter-spacing:-0.03em;">Auto Resume Sifter</div>
  <div style="font-size:0.85rem;color:rgba(255,255,255,0.5);margin-bottom:1.75rem;">
    Intelligent resume screening &amp; bias-aware candidate ranking
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1px;
              background:#C8970A;border:1px solid #C8970A;">
    <div style="background:#0c1a2e;padding:0.9rem 1rem;text-align:left;">
      <div style="font-size:0.75rem;font-weight:700;color:#C8970A;margin-bottom:0.25rem;">TF-IDF Scoring</div>
      <div style="color:rgba(255,255,255,0.45);font-size:0.7rem;line-height:1.5;">
        Cosine-similarity ranking with normalised batch scoring</div>
    </div>
    <div style="background:#0c1a2e;padding:0.9rem 1rem;text-align:left;">
      <div style="font-size:0.75rem;font-weight:700;color:#C8970A;margin-bottom:0.25rem;">Weighted Rubric</div>
      <div style="color:rgba(255,255,255,0.45);font-size:0.7rem;line-height:1.5;">
        Structured JD with skill coverage, education floors &amp; knockout rules</div>
    </div>
    <div style="background:#0c1a2e;padding:0.9rem 1rem;text-align:left;">
      <div style="font-size:0.75rem;font-weight:700;color:#C8970A;margin-bottom:0.25rem;">Bias Audit</div>
      <div style="color:rgba(255,255,255,0.45);font-size:0.7rem;line-height:1.5;">
        Gender · career gap · institution tier · India socio-cultural proxy</div>
    </div>
  </div>
  <div style="margin-top:1.25rem;font-size:0.7rem;color:rgba(255,255,255,0.28);">
    Sign in with your HR account to continue
  </div>
</div>
"""


def navbar_html() -> str:
    """Return the fixed top-navbar background HTML. Page links are added via st.page_link + JS."""
    return (
        '<nav class="ars-nav">'
        '<div class="ars-nav-brand">Auto Resume <em>Sifter</em></div>'
        '</nav>'
    )


def navbar_js(ready: bool = False) -> str:
    """JS (for components.html) that moves st.page_link() elements into the fixed navbar."""
    rdy = "true" if ready else "false"
    return f"""<script>
(function(){{
  var rdy={rdy};
  function tag(){{
    var doc=window.parent.document;
    var els=doc.querySelectorAll('[data-testid="stPageLink"]');
    var i=0;
    els.forEach(function(el){{
      if(i>=3)return;
      if(!el.classList.contains('ars-nav-link'))el.classList.add('ars-nav-link');
      var nc='ars-nl-'+i;
      if(!el.classList.contains(nc))el.classList.add(nc);
      if(!rdy&&i>0&&!el.classList.contains('ars-nl-dim'))el.classList.add('ars-nl-dim');
      if(rdy&&el.classList.contains('ars-nl-dim'))el.classList.remove('ars-nl-dim');
      i++;
    }});
  }}
  var p=window.parent;
  if(!p._arsObs){{
    var t;
    p._arsObs=new p.MutationObserver(function(){{clearTimeout(t);t=setTimeout(tag,60);}});
    p._arsObs.observe(p.document.body,{{childList:true,subtree:true}});
  }}
  tag();
}})();
</script>"""


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _tier_badge(tier: str) -> str:
    cls = {
        "Strong Match": "badge-strong",
        "Partial Match": "badge-partial",
        "Not Suitable": "badge-unsuitable",
    }.get(tier, "")
    return f'<span class="badge {cls}">{tier}</span>'


def _skill_tags(skills: list[str], cls: str = "skill-tag") -> str:
    if not skills:
        return '<span style="color:#8a9ab0;font-size:0.75rem;">—</span>'
    return "".join(f'<span class="{cls}">{s}</span>' for s in skills[:8])


def _html_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a plain HTML <table> — no Arrow/JS runtime required."""
    if df.empty:
        return '<p style="color:#8a9ab0;font-size:0.85rem;margin:0.4rem 0;">No data.</p>'
    headers = "".join(f"<th>{col}</th>" for col in df.columns)
    rows = ""
    for _, row in df.iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row)
        rows += f"<tr>{cells}</tr>"
    return (
        '<div style="overflow-x:auto;margin-top:0.5rem;">'
        '<table class="results-table" style="min-width:100%;font-size:0.82rem;">'
        f'<thead><tr>{headers}</tr></thead>'
        f'<tbody>{rows}</tbody>'
        '</table></div>'
    )


def _build_csv(candidates) -> bytes:
    rows = [
        {
            "Rank": i,
            "Filename": c.filename,
            "Name": c.name or "",
            "Raw Score (%)": c.raw_score,
            "Normalised Score (%)": c.score,
            "Tier": c.tier,
            "Matched Skills": ", ".join(c.matched_skills),
        }
        for i, c in enumerate(candidates, 1)
    ]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _disp_score(c) -> float:
    """Return final_score when a rubric was applied, else fall back to normalised score."""
    return c.final_score if c.final_score > 0 else c.score


def _gap_recommendation(c, jd_skills: list[str]) -> str:
    matched_lower = {s.lower() for s in c.matched_skills}
    missing = [s for s in jd_skills if s.lower() not in matched_lower]
    coverage = len(c.matched_skills) / max(len(jd_skills), 1) * 100
    if coverage >= 70:
        return "Broad skill coverage — consider progressing to a technical assessment."
    if missing and coverage >= 40:
        top_missing = ", ".join(missing[:3])
        return (
            f"Solid background but missing key requirements: {top_missing}. "
            "Consider a targeted skills interview."
        )
    return "Limited overlap with JD requirements. May suit a different or more junior role."


def _plotly_layout(**kwargs) -> dict:
    base = dict(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", size=12, color="#1a2940"),
        margin=dict(t=30, b=50, l=50, r=20),
        height=360,
    )
    base.update(kwargs)
    return base