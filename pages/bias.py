import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.classifier import Candidate
from src.ui_helpers import _html_table, _disp_score, _plotly_layout

# ── Guard ─────────────────────────────────────────────────────────────────────
if not st.session_state.get("ready"):
    st.markdown(
        '<div class="section-title">Bias Audit</div>'
        '<div class="section-sub">Run an analysis first to see the bias audit here.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-box" style="margin-top:1rem;">'
        'Go to <a href="/upload" style="color:#1E3A5F;font-weight:600;">Upload &amp; Analyse</a> '
        'to load demo data or upload your own resumes.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ── Load session state ─────────────────────────────────────────────────────────
ranked: list[Candidate] = st.session_state["ranked"]
audit = st.session_state["audit"]

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-title">Bias Audit</div>'
    '<div class="section-sub">'
    'Multi-dimensional fairness analysis: gender · career gap · institution tier · '
    'India-specific socio-cultural surname proxy.</div>',
    unsafe_allow_html=True,
)


# ── Helper: render a dimension stats table ────────────────────────────────────
def _render_dim_stats(dim_result) -> None:
    if not dim_result.audited:
        st.markdown(
            f'<div class="info-box"><strong>Skipped.</strong> {dim_result.skip_reason}</div>',
            unsafe_allow_html=True,
        )
        return
    if dim_result.insufficient_data or not dim_result.group_stats:
        st.markdown(
            '<div class="info-box">Insufficient data — need at least 3 candidates per group '
            'for a reliable statistical test.</div>',
            unsafe_allow_html=True,
        )
        return
    if dim_result.bias_detected:
        _correction_note = (
            " Automated correction applied." if dim_result.corrected
            else " Advisory only — no auto-correction."
        )
        st.markdown(
            f'<div class="bias-warn"><strong>Disparity detected.</strong> '
            f'pp-gap: {dim_result.pp_gap:.1f} pp · DIR: {dim_result.dir_value:.2f} · '
            f'{dim_result.test_name} p={dim_result.p_value}.{_correction_note}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bias-clean"><strong>No significant disparity.</strong> '
            f'pp-gap: {dim_result.pp_gap:.1f} pp · DIR: {dim_result.dir_value:.2f} · '
            f'{dim_result.test_name} p={dim_result.p_value}</div>',
            unsafe_allow_html=True,
        )
    if dim_result.low_sample_groups:
        st.markdown(
            f'<div class="warn-box" style="margin-top:0.4rem;">Low sample: '
            f'{", ".join(dim_result.low_sample_groups)} — estimates unreliable.</div>',
            unsafe_allow_html=True,
        )
    _rows = [
        {
            "Group": g,
            "Candidates": s["count"],
            "Mean Score (%)": s["mean_score"],
            "Low Sample": "⚠" if g in dim_result.low_sample_groups else "—",
        }
        for g, s in dim_result.group_stats.items()
    ]
    if _rows:
        st.markdown(_html_table(pd.DataFrame(_rows)), unsafe_allow_html=True)


# ── Step 1: What was checked ──────────────────────────────────────────────────
st.markdown("#### Step 1 — What We Checked")
_dr = audit.dimension_results
_step1_dims = [
    ("Gender", _dr.get("gender")),
    ("Career Gap", _dr.get("career_gap")),
    ("Institution Tier", _dr.get("institution_tier")),
    ("Socio-cultural", _dr.get("socio_cultural")),
]
_dim_cols = st.columns(4)
for _dc, (_dlabel, _ddim) in zip(_dim_cols, _step1_dims):
    with _dc:
        _audited = _ddim.audited if _ddim else True
        _bias = _ddim.bias_detected if _ddim else False
        _dcolor = "#8b1a1a" if _bias else ("#1E3A5F" if _audited else "#8a9ab0")
        _dbadge = "Bias Detected" if _bias else ("Active" if _audited else "Skipped")
        _dbadge_cls = "badge-unsuitable" if _bias else ("badge-strong" if _audited else "badge-partial")
        _dreason = (
            _ddim.skip_reason if (_ddim and not _ddim.audited)
            else (f"pp-gap {_ddim.pp_gap:.0f}" if (_ddim and _ddim.group_stats) else "Active")
        )
        st.markdown(
            f'<div class="stat-card" style="border-top-color:{_dcolor};">'
            f'<div class="stat-label">{_dlabel}</div>'
            f'<div style="margin-top:0.4rem;"><span class="badge {_dbadge_cls}">{_dbadge}</span></div>'
            f'<div style="font-size:0.65rem;color:#8a9ab0;margin-top:0.3rem;">{str(_dreason)[:55]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Step 2: What We Found ─────────────────────────────────────────────────────
st.markdown("#### Step 2 — What We Found")

with st.expander("Gender disparity", expanded=True):
    if "gender" in _dr:
        _render_dim_stats(_dr["gender"])
    if audit.inference_rows:
        with st.expander("Per-candidate gender inference"):
            st.markdown(_html_table(
                pd.DataFrame(audit.inference_rows).rename(columns={
                    "filename": "File", "name": "Name",
                    "inferred_gender": "Inferred Gender", "method": "Method",
                })
            ), unsafe_allow_html=True)

with st.expander("Career gap disparity"):
    if "career_gap" in _dr:
        _render_dim_stats(_dr["career_gap"])
    if "career_gap" in _dr and _dr["career_gap"].audited:
        _gap_rows = [
            {
                "Candidate": c.name or c.filename,
                "Career Gap (months)": c.career_gap_months,
                "Score (%)": _disp_score(c),
            }
            for c in ranked
        ]
        if _gap_rows:
            st.markdown(_html_table(pd.DataFrame(_gap_rows)), unsafe_allow_html=True)

with st.expander("Institution tier disparity"):
    if "institution_tier" in _dr:
        _render_dim_stats(_dr["institution_tier"])
    if "institution_tier" in _dr and _dr["institution_tier"].audited:
        _inst_rows = [
            {
                "Candidate": c.name or c.filename,
                "Institution Tier": c.institution_tier,
                "Score (%)": _disp_score(c),
            }
            for c in ranked
        ]
        if _inst_rows:
            st.markdown(_html_table(pd.DataFrame(_inst_rows)), unsafe_allow_html=True)

with st.expander("Socio-cultural surname proxy (India)"):
    st.markdown(
        '<div class="warn-box" style="margin-bottom:0.6rem;">'
        '<strong>Disclaimer:</strong> This analysis uses surname-based proxies for region, '
        'religion, and caste group. These are statistical approximations — may be incorrect '
        'for individual candidates. Purpose: detect systemic scoring patterns only.</div>',
        unsafe_allow_html=True,
    )
    _socio_dim = _dr.get("socio_cultural")
    _sub_dims = _socio_dim.extra.get("sub_dimensions", {}) if _socio_dim else {}
    for _sub_key, _sub_label in [
        ("region_proxy", "Region"),
        ("religion_proxy", "Religion"),
        ("caste_group_proxy", "Caste Group"),
    ]:
        _sub = _sub_dims.get(_sub_key)
        if _sub:
            st.markdown(f"**{_sub_label}**")
            _render_dim_stats(_sub)
            if _sub_key == "caste_group_proxy" and _sub.bias_detected:
                st.markdown(
                    '<div class="info-box" style="margin-top:0.4rem;">'
                    '<strong>Context:</strong> Caste-based disparities are constitutionally '
                    'significant in India. SC/ST/OBC candidates are protected under Articles 15 '
                    'and 16. If this system is used for employment decisions, verify compliance '
                    'with applicable reservation policies.</div>',
                    unsafe_allow_html=True,
                )
    _proxy_rows = [
        {
            "Candidate": c.name or c.filename,
            "Region": c.region_proxy,
            "Religion": c.religion_proxy,
            "Caste Group": c.caste_group_proxy,
            "Score (%)": _disp_score(c),
        }
        for c in ranked
    ]
    with st.expander("Per-candidate proxy lookup"):
        st.markdown(_html_table(pd.DataFrame(_proxy_rows)), unsafe_allow_html=True)

# ── Step 3: What We Corrected ─────────────────────────────────────────────────
st.markdown("#### Step 3 — What We Corrected")
if audit.corrected and audit.candidate_deltas:
    st.markdown(
        '<p style="font-size:0.85rem;color:#5e738a;margin-bottom:0.5rem;">'
        'Name-masking counterfactual correction was applied. '
        'Final score = average of original TF-IDF score and name-masked re-score.</p>',
        unsafe_allow_html=True,
    )
    _delta_df = pd.DataFrame(audit.candidate_deltas).rename(columns={
        "filename": "File", "name": "Name",
        "original_score": "Original (%)",
        "masked_score": "Name-masked (%)",
        "corrected_score": "Corrected (%)",
    })
    st.markdown(_html_table(_delta_df), unsafe_allow_html=True)

    _ba_names = [d["name"] or d["filename"] for d in audit.candidate_deltas]
    _fig_ba = go.Figure()
    _fig_ba.add_trace(go.Bar(
        name="Original",
        y=_ba_names, x=[d["original_score"] for d in audit.candidate_deltas],
        orientation="h", marker_color="#9ab0c8", marker_line_width=0,
        hovertemplate="%{y}<br>Original: %{x:.1f}%<extra></extra>",
    ))
    _fig_ba.add_trace(go.Bar(
        name="Corrected",
        y=_ba_names, x=[d["corrected_score"] for d in audit.candidate_deltas],
        orientation="h", marker_color="#1E3A5F", marker_line_width=0,
        hovertemplate="%{y}<br>Corrected: %{x:.1f}%<extra></extra>",
    ))
    _fig_ba.update_layout(**_plotly_layout(
        barmode="group",
        xaxis=dict(title="Score (%)", range=[0, 115], showgrid=True, gridcolor="#f0f4f8"),
        yaxis=dict(title=None, showgrid=False, autorange="reversed"),
        legend=dict(orientation="h", y=1.05),
        height=max(280, len(_ba_names) * 42 + 80),
        margin=dict(t=30, b=40, l=160, r=60),
    ))
    st.plotly_chart(_fig_ba, width="stretch", key="chart_correction_ba")
else:
    st.markdown(
        '<div class="bias-clean">No corrections applied — '
        'no statistically significant gender disparity detected.</div>',
        unsafe_allow_html=True,
    )

# ── Step 4: Score distribution by gender ──────────────────────────────────────
st.markdown("#### Step 4 — Score Distribution by Gender")
_chart_data = [
    {"name": c.name or c.filename, "score": _disp_score(c),
     "gender": getattr(c, "_gender", "Unknown")}
    for c in ranked
]
_gender_colors = {"Male": "#1E3A5F", "Female": "#C8970A", "Unknown": "#8a9ab0"}
_fig_gender = go.Figure()
for _g in ["Male", "Female", "Unknown"]:
    _subset = [d for d in _chart_data if d["gender"] == _g]
    if not _subset:
        continue
    _fig_gender.add_trace(go.Bar(
        y=[d["name"] for d in _subset],
        x=[d["score"] for d in _subset],
        orientation="h", name=_g,
        marker_color=_gender_colors[_g], marker_line_width=0,
        hovertemplate="%{y}<br>Score: %{x:.1f}%<extra></extra>",
    ))
_fig_gender.update_layout(**_plotly_layout(
    barmode="group",
    xaxis=dict(title="Score (%)", range=[0, 115], showgrid=True, gridcolor="#f0f4f8"),
    yaxis=dict(title=None, showgrid=False, autorange="reversed"),
    legend=dict(orientation="h", y=1.05, title=None),
    height=max(280, len(ranked) * 28 + 80),
    margin=dict(t=30, b=40, l=160, r=60),
))
st.plotly_chart(_fig_gender, width="stretch", key="chart_gender_dist")

# ── Methodology expander ──────────────────────────────────────────────────────
with st.expander("Audit methodology & limitations"):
    st.markdown("""
**Scoring**
TF-IDF cosine similarity scores are content-based. Names alone rarely influence the score.
The audit primarily verifies this expectation holds in each batch.

**Gender audit** (conditional)
Active only when JD gender preference = "No preference". Compares mean scores across inferred
gender groups; flags if any two known groups differ by > 10 percentage points.
Counterfactual correction: name → `[CANDIDATE]` (all occurrences), re-score,
average of original + masked score. Up to 3 iterative rounds until delta < 2 pp.

**Career gap audit** (conditional)
Active only when JD max gap = 0 (not restricted). Detects if candidates with employment gaps
score systematically lower than those without — an unintended penalty.

**Institution tier**
Matches institution names from resume text against a curated top-tier list.
Flags if tier-based score disparities exist.

**Socio-cultural proxy (India)**
Surname lookup against a regional database mapping last names to
region (North/South/East/West), religion, and caste group (General/OBC/SC/ST).
Clearly labelled as approximate — intended for systemic pattern detection, not
individual assessment. Caste disparity results reference India's constitutional
reservation framework (SC/ST/OBC protections) for context.

**Limitations**
Statistical reliability requires n > 30 per group. At demo scale, treat findings as
indicative. A production deployment should use Fairlearn or AI Fairness 360 with
ongoing monitoring.
    """)

st.markdown(
    '<div class="note-box" style="margin-top:1rem;">'
    '<strong>Note on bias correction:</strong> Automated correction is applied only for gender '
    '(name-masking counterfactual). Career gap, institution tier, and socio-cultural '
    'dimensions show advisories only — those signals may be intentional job criteria depending on context.</div>',
    unsafe_allow_html=True,
)
