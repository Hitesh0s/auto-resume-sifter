import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.classifier import Candidate
from src.scorer import top_tfidf_terms
from src.ui_helpers import (
    _tier_badge, _skill_tags, _html_table,
    _build_csv, _disp_score, _gap_recommendation, _plotly_layout,
)

# ── Guard ─────────────────────────────────────────────────────────────────────
if not st.session_state.get("ready"):
    st.markdown(
        '<div class="section-title">Results</div>'
        '<div class="section-sub">Run an analysis first to see results here.</div>',
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
jd_skills: list[str] = st.session_state.get("jd_skills", [])
_vectorizer = st.session_state.get("vectorizer")
_tfidf_matrix = st.session_state.get("tfidf_matrix")
_tfidf_order: list[str] = st.session_state.get("tfidf_order", [])
_tfidf_row_by_filename: dict[str, int] = {fn: i + 1 for i, fn in enumerate(_tfidf_order)}
_rubric_active = st.session_state.get("rubric") is not None

top_disp_score = max((_disp_score(c) for c in ranked), default=0.0)

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-title">Candidate Results</div>'
    '<div class="section-sub">'
    'Ranked by final score (TF-IDF × 70% + rubric skill coverage × 30%). '
    'Strong ≥ 70 · Partial ≥ 40 · Not Suitable &lt; 40.</div>',
    unsafe_allow_html=True,
)

# ── Stat cards ────────────────────────────────────────────────────────────────
n_strong = sum(1 for c in ranked if c.tier == "Strong Match")
n_partial = sum(1 for c in ranked if c.tier == "Partial Match")
n_unsuitable = sum(1 for c in ranked if c.tier == "Not Suitable")
avg_score = round(sum(_disp_score(c) for c in ranked) / max(len(ranked), 1), 1)

sc1, sc2, sc3, sc4, sc5 = st.columns(5)
for _col, _label, _val, _color in [
    (sc1, "Total", len(ranked), "#1E3A5F"),
    (sc2, "Strong Match", n_strong, "#155724"),
    (sc3, "Partial Match", n_partial, "#7d5a00"),
    (sc4, "Not Suitable", n_unsuitable, "#8b1a1a"),
    (sc5, "Avg Score", f"{avg_score}%", "#1E3A5F"),
]:
    with _col:
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">{_label}</div>'
            f'<div class="stat-value" style="color:{_color};">{_val}</div></div>',
            unsafe_allow_html=True,
        )

# ── Score formula expander ────────────────────────────────────────────────────
with st.expander("How final scores are calculated", expanded=False):
    if _rubric_active:
        st.markdown("""
**Final Score = (TF-IDF normalised score × 70%) + (Rubric skill coverage × 30%)**

| Component | Weight | What it measures |
|---|---|---|
| **TF-IDF normalised score** | 70% | Cosine similarity vs JD, normalised so top = 100% |
| **Rubric skill coverage** | 30% | Fraction of required skills found in the resume |
| **Knockout** | override | Degree/CGPA floor not met → score forced to 0 |
""")
    else:
        st.markdown("""
**Final Score = TF-IDF normalised score (no rubric active)**

| Component | What it measures |
|---|---|
| **TF-IDF raw score** | Cosine similarity × 100 (typically 5–45%) |
| **Normalised score** | Raw ÷ batch maximum × 100 — top candidate always = 100% |

Use the **Structured Form** tab on the Upload page to activate the 70/30 rubric blend.
""")
    if ranked:
        _breakdown_rows = []
        for c in ranked[:10]:
            rr = c.rubric_result
            tfidf_contrib = round(c.score * 0.70, 1) if _rubric_active else None
            rubric_contrib = round((rr.bonus * 100) * 0.30, 1) if (rr and not rr.knockout) else None
            _breakdown_rows.append({
                "Candidate": c.name or c.filename,
                "TF-IDF (norm %)": f"{c.score}%",
                **({"TF-IDF × 0.70": f"{tfidf_contrib}%",
                    "Rubric × 0.30": f"{rubric_contrib}%" if rubric_contrib is not None
                    else ("KO" if rr and rr.knockout else "—"),
                    } if _rubric_active else {}),
                "Final Score": f"{_disp_score(c):.1f}%",
                "Tier": c.tier,
            })
        st.markdown(
            '<p style="font-size:0.78rem;color:#5e738a;margin:0.75rem 0 0.25rem;">'
            'Top 10 candidates breakdown:</p>',
            unsafe_allow_html=True,
        )
        st.markdown(_html_table(pd.DataFrame(_breakdown_rows)), unsafe_allow_html=True)

# ── Top Scorer Spotlight ──────────────────────────────────────────────────────
if ranked:
    _top = ranked[0]
    _top_terms: list[tuple[str, float]] = []
    if _vectorizer is not None and _tfidf_matrix is not None:
        _top_row = _tfidf_row_by_filename.get(_top.filename)
        if _top_row is not None:
            _top_terms = top_tfidf_terms(_vectorizer, _tfidf_matrix, _top_row, top_n=5)
    _top_skills_html = _skill_tags(_top.matched_skills[:6], "skill-tag-green") or ""
    _top_terms_html = "".join(
        f'<span class="skill-tag" style="background:#fff8e6;color:#7d5a00;border-color:#C8970A;">{t}</span>'
        for t, _ in _top_terms
    )
    st.markdown(
        f'<div class="top-scorer-card">'
        f'<div style="display:flex;align-items:flex-start;gap:1.5rem;flex-wrap:wrap;">'
        f'<div>'
        f'  <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;'
        f'    letter-spacing:0.1em;color:#C8970A;margin-bottom:0.2rem;">Top Scorer</div>'
        f'  <div style="font-size:1.2rem;font-weight:700;color:#1E3A5F;">'
        f'    {_top.name or _top.filename}</div>'
        f'  <div style="font-size:0.78rem;color:#5e738a;margin-top:0.1rem;">{_top.filename}</div>'
        f'</div>'
        f'<div style="margin-left:auto;text-align:right;">'
        f'  <div style="font-size:2rem;font-weight:800;color:#C8970A;line-height:1;">'
        f'    {_disp_score(_top):.0f}%</div>'
        f'  <div style="font-size:0.7rem;color:#5e738a;">final score</div>'
        f'  <div style="font-size:0.72rem;color:#8a9ab0;margin-top:0.15rem;">'
        f'    TF-IDF: {_top.raw_score}%</div>'
        f'</div>'
        f'</div>'
        f'<div style="margin-top:0.75rem;font-size:0.75rem;color:#475569;font-weight:600;">'
        f'Matched skills:</div>'
        f'<div style="margin-top:0.2rem;">{_top_skills_html}</div>'
        + (
            f'<div style="margin-top:0.5rem;font-size:0.75rem;color:#475569;font-weight:600;">'
            f'Top TF-IDF terms:</div>'
            f'<div style="margin-top:0.2rem;">{_top_terms_html}</div>'
            if _top_terms else ""
        )
        + '</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="margin-top:1.25rem;"></div>', unsafe_allow_html=True)

# ── Tier filter + Results Table ───────────────────────────────────────────────
tier_filter = st.selectbox(
    "Filter by tier",
    ["All", "Strong Match", "Partial Match", "Not Suitable"],
)
filtered = ranked if tier_filter == "All" else [c for c in ranked if c.tier == tier_filter]


def _build_results_html(candidates: list[Candidate], top_score: float) -> str:
    rows = ""
    for i, c in enumerate(candidates, 1):
        ds = _disp_score(c)
        score_cls = "score-cell top-score" if ds == top_score else "score-cell"
        rubric_ko = c.rubric_result is not None and c.rubric_result.knockout
        ko_badge = (
            ' <span class="badge badge-unsuitable" style="font-size:0.62rem;">Knockout</span>'
            if rubric_ko else ""
        )
        rows += (
            f"<tr>"
            f"<td style='color:#8a9ab0;'>{i}</td>"
            f"<td><strong>{c.name or c.filename}</strong>"
            f"<br><span style='font-size:0.73rem;color:#8a9ab0;'>{c.filename}</span></td>"
            f"<td class='score-cell' style='color:#5e738a;'>{c.raw_score}%</td>"
            f"<td class='{score_cls}'>{ds:.1f}%{ko_badge}</td>"
            f"<td>{_tier_badge(c.tier)}</td>"
            f"<td>{_skill_tags(c.matched_skills)}</td>"
            f"</tr>"
        )
    return (
        '<table class="results-table">'
        "<thead><tr>"
        "<th>#</th><th>Candidate</th><th>TF-IDF Raw</th>"
        "<th>Final Score</th><th>Tier</th><th>Matched Skills</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
    )


st.markdown(_build_results_html(filtered, top_disp_score), unsafe_allow_html=True)

# ── Score Analytics expander ──────────────────────────────────────────────────
with st.expander("Score Analytics — Algorithm Insights", expanded=True):
    _anal_tab1, _anal_tab2, _anal_tab3 = st.tabs([
        "Score Distribution", "TF-IDF Term Contribution", "Skill Coverage Heatmap",
    ])

    with _anal_tab1:
        st.markdown(
            '<p style="font-size:0.8rem;color:#5e738a;margin-bottom:0.5rem;">'
            'Where candidates cluster across the normalised score range. '
            'Coloured bands show tier thresholds.</p>',
            unsafe_allow_html=True,
        )
        _scores_disp = [_disp_score(c) for c in ranked]
        _names_disp = [c.name or c.filename for c in ranked]
        _colors_tier = [
            "#155724" if c.tier == "Strong Match"
            else "#7d5a00" if c.tier == "Partial Match"
            else "#8b1a1a"
            for c in ranked
        ]
        _fig_dist = go.Figure()
        _fig_dist.add_vrect(x0=0, x1=40, fillcolor="#fce9e9", layer="below", line_width=0,
                            annotation_text="Not Suitable", annotation_position="top left",
                            annotation_font=dict(size=10, color="#8b1a1a"))
        _fig_dist.add_vrect(x0=40, x1=70, fillcolor="#fff8e6", layer="below", line_width=0,
                            annotation_text="Partial", annotation_position="top left",
                            annotation_font=dict(size=10, color="#7d5a00"))
        _fig_dist.add_vrect(x0=70, x1=100, fillcolor="#e8f5ec", layer="below", line_width=0,
                            annotation_text="Strong", annotation_position="top left",
                            annotation_font=dict(size=10, color="#155724"))
        _fig_dist.add_trace(go.Bar(
            x=_scores_disp, y=_names_disp, orientation="h",
            marker_color=_colors_tier, marker_line_width=0,
            text=[f"{s:.0f}%" for s in _scores_disp], textposition="outside",
            hovertemplate="%{y}<br>Score: %{x:.1f}%<extra></extra>",
        ))
        _fig_dist.update_layout(**_plotly_layout(
            xaxis=dict(title="Final Score (%)", range=[0, 115], showgrid=True, gridcolor="#f0f4f8"),
            yaxis=dict(autorange="reversed", title=None, showgrid=False),
            height=max(300, len(ranked) * 28),
            margin=dict(t=20, b=40, l=160, r=70),
            showlegend=False,
        ))
        st.plotly_chart(_fig_dist, width="stretch", key="chart_score_dist")

    with _anal_tab2:
        st.markdown(
            '<p style="font-size:0.8rem;color:#5e738a;margin-bottom:0.5rem;">'
            'The TF-IDF algorithm weights rare-but-present terms highly. '
            'Select a candidate to see the 10 terms that contributed most to their score.</p>',
            unsafe_allow_html=True,
        )
        if _vectorizer is not None and _tfidf_matrix is not None:
            _cand_names = [c.name or c.filename for c in ranked]
            _sel_cand = st.selectbox("Candidate", _cand_names, key="tfidf_cand_sel")
            _sel_idx = _cand_names.index(_sel_cand)
            _sel_row = _tfidf_row_by_filename.get(ranked[_sel_idx].filename)
            _terms = (
                top_tfidf_terms(_vectorizer, _tfidf_matrix, _sel_row, top_n=10)
                if _sel_row is not None else []
            )
            if _terms:
                _term_names = [t for t, _ in _terms][::-1]
                _term_vals = [v for _, v in _terms][::-1]
                _fig_tfidf = go.Figure(go.Bar(
                    x=_term_vals, y=_term_names, orientation="h",
                    marker_color="#1E3A5F", marker_line_width=0,
                    text=[f"{v:.4f}" for v in _term_vals], textposition="outside",
                    hovertemplate="Term: %{y}<br>TF-IDF: %{x:.4f}<extra></extra>",
                ))
                _fig_tfidf.update_layout(**_plotly_layout(
                    xaxis=dict(title="TF-IDF weight", showgrid=True, gridcolor="#f0f4f8"),
                    yaxis=dict(title=None, showgrid=False),
                    height=340, margin=dict(t=20, b=40, l=140, r=60),
                ))
                st.plotly_chart(_fig_tfidf, width="stretch", key="chart_tfidf_terms")
                st.markdown(
                    '<div class="note-box">Terms from this candidate\'s resume with the highest '
                    'TF-IDF weight against the JD. Terms rare across the corpus but present here '
                    'score highest.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("No significant terms found.")
        else:
            st.info("TF-IDF data not available for this session.")

    with _anal_tab3:
        st.markdown(
            '<p style="font-size:0.8rem;color:#5e738a;margin-bottom:0.5rem;">'
            'Skill coverage per candidate across matched skill categories. '
            'Darker = more skills from that category found.</p>',
            unsafe_allow_html=True,
        )
        if jd_skills:
            _rubric_data = st.session_state.get("rubric") or {}
            _skill_cats: dict[str, list[str]] = _rubric_data.get("skills", {})
            if not _skill_cats:
                _skill_cats = {"All JD Skills": jd_skills}
            _top10 = ranked[:10]
            _cand_labels = [c.name or c.filename for c in _top10]
            _cat_labels = list(_skill_cats.keys())
            _z: list[list[float]] = []
            for _cat, _cat_skills in _skill_cats.items():
                _row: list[float] = []
                for c in _top10:
                    _cmatched = {s.lower() for s in c.matched_skills}
                    _cov = sum(1 for s in _cat_skills if s.lower() in _cmatched) / max(len(_cat_skills), 1)
                    _row.append(round(_cov * 100, 1))
                _z.append(_row)
            _fig_heat = go.Figure(go.Heatmap(
                z=_z, x=_cand_labels, y=_cat_labels,
                colorscale=[[0, "#f0f4f8"], [0.5, "#7fa9d0"], [1, "#1E3A5F"]],
                text=[[f"{v:.0f}%" for v in row] for row in _z],
                texttemplate="%{text}",
                hovertemplate="Category: %{y}<br>Candidate: %{x}<br>Coverage: %{text}<extra></extra>",
                showscale=True, zmin=0, zmax=100,
            ))
            _fig_heat.update_layout(**_plotly_layout(
                xaxis=dict(tickangle=-25, title=None),
                yaxis=dict(title=None),
                height=max(260, len(_cat_labels) * 50 + 80),
                margin=dict(t=20, b=80, l=160, r=20),
            ))
            st.plotly_chart(_fig_heat, width="stretch", key="chart_skill_heatmap")
        else:
            st.info("Skill heatmap requires JD skills — use the Structured Form or paste a skill-rich JD.")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Gap Analysis ──────────────────────────────────────────────────────────────
partial_matches = [c for c in ranked if c.tier == "Partial Match"]
if partial_matches:
    st.markdown(
        '<div class="section-title">Gap Analysis</div>'
        '<div class="section-sub">Partial Match candidates — expand each to see matched vs missing skills.</div>',
        unsafe_allow_html=True,
    )
    for c in partial_matches:
        matched_lower = {s.lower() for s in c.matched_skills}
        missing_skills = [s for s in jd_skills if s.lower() not in matched_lower]
        ds = _disp_score(c)
        label = f"{c.name or c.filename}  —  {ds:.0f}% match"
        with st.expander(label):
            st.progress(ds / 100)
            gap_col1, gap_col2 = st.columns(2)
            with gap_col1:
                st.markdown(
                    '<p style="font-weight:600;font-size:0.85rem;color:#155724;margin-bottom:0.3rem;">'
                    '&#10003; Skills found</p>', unsafe_allow_html=True,
                )
                st.markdown(
                    _skill_tags(c.matched_skills, "skill-tag-green") if c.matched_skills
                    else '<span style="color:#8a9ab0;font-size:0.8rem;">None detected</span>',
                    unsafe_allow_html=True,
                )
            with gap_col2:
                st.markdown(
                    '<p style="font-weight:600;font-size:0.85rem;color:#7d5a00;margin-bottom:0.3rem;">'
                    '&#10007; Skills from JD not found</p>', unsafe_allow_html=True,
                )
                st.markdown(
                    _skill_tags(missing_skills, "skill-tag-orange") if missing_skills
                    else '<span style="color:#155724;font-size:0.8rem;">All JD skills covered</span>',
                    unsafe_allow_html=True,
                )
            rec = _gap_recommendation(c, jd_skills)
            st.markdown(
                f'<div class="note-box" style="margin-top:0.75rem;">'
                f'<strong>Recommendation:</strong> {rec}</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── CSV Export ────────────────────────────────────────────────────────────────
st.download_button(
    label="Export CSV",
    data=_build_csv(ranked),
    file_name="auto_resume_sifter_results.csv",
    mime="text/csv",
)