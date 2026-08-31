import os
import pathlib
import tempfile
import tkinter as tk
from tkinter import filedialog

import streamlit as st

from src.reader import extract_text
from src.preprocessor import preprocess
from src.extractor import extract_name, extract_skills
from src.scorer import score_batch
from src.classifier import Candidate, apply_final_score, rank_candidates
from src.bias_audit import run_audit
from src.jd_builder import (
    PROFESSION_SKILLS, SOFT_SKILLS_COMMON, SENIORITY_LEVELS,
    WORK_MODES, DEGREE_REQUIREMENTS, GENDER_OPTIONS, build_jd,
)
from src.rubric_scorer import apply_rubric

_DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
_DEMO_RESUMES_DIR = _DATA_DIR / "demo_resumes"
_DEMO_JD_CANDIDATES = [
    _DEMO_RESUMES_DIR / "jd.pdf",
    _DEMO_RESUMES_DIR / "jd.txt",
    _DATA_DIR / "demo_jd.pdf",
    _DATA_DIR / "demo_jd.txt",
]
_DEMO_JD_PATH = next((p for p in _DEMO_JD_CANDIDATES if p.exists()), None)


def _pick_folder() -> str:
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    folder = filedialog.askdirectory(title="Select folder containing resumes")
    root.destroy()
    return folder or ""


# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-title">Upload &amp; Analyse</div>'
    '<div class="section-sub">Build a job description, upload candidate resumes, '
    'and run the screening pipeline.</div>',
    unsafe_allow_html=True,
)

# ── Demo banner ────────────────────────────────────────────────────────────────
if _DEMO_RESUMES_DIR.exists():
    _demo_files = [
        p for p in
        sorted(_DEMO_RESUMES_DIR.glob("*.pdf")) + sorted(_DEMO_RESUMES_DIR.glob("*.txt"))
        if p.stem != "jd"
    ]
else:
    _demo_files = []
_demo_ready = _DEMO_JD_PATH is not None and len(_demo_files) > 0

if _demo_ready:
    _dbc1, _dbc2 = st.columns([5, 1])
    with _dbc1:
        st.markdown(
            f'<div class="demo-banner">'
            f'<div style="flex:1;">'
            f'<div style="font-size:0.6rem;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:#C8970A;margin-bottom:0.2rem;">Quick Demo</div>'
            f'<div style="color:#fff;font-size:0.88rem;font-weight:600;">'
            f'{len(_demo_files)} sample resumes ready'
            f'<span style="color:rgba(255,255,255,0.5);font-weight:400;margin-left:0.5rem;">·</span>'
            f'<span style="color:rgba(255,255,255,0.65);font-weight:400;margin-left:0.5rem;">'
            f'Software Engineer JD</span></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with _dbc2:
        _load_demo = st.button("Load Demo Data", use_container_width=True)

    if _load_demo:
        with st.spinner("Running demo analysis…"):
            if _DEMO_JD_PATH.suffix.lower() == ".pdf":
                jd_raw = extract_text(str(_DEMO_JD_PATH))
            else:
                jd_raw = _DEMO_JD_PATH.read_text(encoding="utf-8")
            jd_preprocessed = preprocess(jd_raw)
            jd_skills = extract_skills(jd_raw)
            demo_candidates: list[Candidate] = []
            for fpath in _demo_files:
                try:
                    raw = extract_text(str(fpath))
                except Exception:
                    continue
                demo_candidates.append(Candidate(
                    filename=fpath.name,
                    name=extract_name(raw),
                    raw_text=raw,
                    preprocessed_text=preprocess(raw),
                    matched_skills=extract_skills(raw),
                ))
            raw_scores, norm_scores, _vec, _mat = score_batch(
                jd_preprocessed,
                [c.preprocessed_text for c in demo_candidates],
                return_vectorizer=True,
            )
            for c, raw_s, norm_s in zip(demo_candidates, raw_scores, norm_scores):
                c.raw_score = raw_s
                c.score = norm_s
                apply_final_score(c)
            ranked = rank_candidates(demo_candidates)
            audit = run_audit(ranked, jd_raw, jd_preprocessed)
            st.session_state.update({
                "ranked": ranked,
                "audit": audit,
                "jd_skills": jd_skills,
                "rubric": None,
                "vectorizer": _vec,
                "tfidf_matrix": _mat,
                "tfidf_order": [c.filename for c in demo_candidates],
                "ready": True,
            })
        st.success("Demo loaded — click **Results** in the nav bar to view rankings.")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── JD Input ──────────────────────────────────────────────────────────────────
st.markdown("### Job Description")
jd_tab_form, jd_tab_text = st.tabs(["Structured Form", "Paste / Upload"])

jd_raw: str = ""
_rubric: dict | None = None

with jd_tab_form:
    st.markdown('<div class="section-heading">Role</div>', unsafe_allow_html=True)
    _fc1, _fc2, _fc3 = st.columns([2, 2, 2])
    with _fc1:
        _profession = st.selectbox(
            "Profession",
            list(PROFESSION_SKILLS.keys()) + ["Custom"],
            key="form_profession",
        )
    with _fc2:
        _seniority = st.selectbox("Seniority", SENIORITY_LEVELS, index=2, key="form_seniority")
    with _fc3:
        _work_mode = st.multiselect("Work Mode", WORK_MODES, default=["Hybrid"], key="form_work_mode")

    _exp_col1, _exp_col2 = st.columns(2)
    with _exp_col1:
        _exp_min = st.slider("Min Experience (years)", 0, 20, 2, key="form_exp_min")
    with _exp_col2:
        _exp_max = st.slider("Max Experience (years)", 0, 30, 6, key="form_exp_max")

    _selected_skills: dict[str, list[str]] = {}
    if _profession != "Custom":
        st.markdown('<div class="section-heading">Skills</div>', unsafe_allow_html=True)
        _skill_map = PROFESSION_SKILLS.get(_profession, {})
        for _cat, _skill_list in _skill_map.items():
            st.markdown(f"**{_cat}**")
            _ncols = 4
            _cols = st.columns(_ncols)
            _cat_selected: list[str] = []
            for _i, _skill in enumerate(_skill_list):
                if _cols[_i % _ncols].checkbox(_skill, key=f"skill_{_profession}_{_cat}_{_skill}"):
                    _cat_selected.append(_skill)
            if _cat_selected:
                _selected_skills[_cat] = _cat_selected

    st.markdown('<div class="section-heading">Soft Skills</div>', unsafe_allow_html=True)
    _soft_cols = st.columns(4)
    _soft_selected: list[str] = []
    for _i, _ss in enumerate(SOFT_SKILLS_COMMON):
        if _soft_cols[_i % 4].checkbox(_ss, key=f"soft_{_ss}"):
            _soft_selected.append(_ss)

    st.markdown('<div class="section-heading">Education Requirements</div>', unsafe_allow_html=True)
    _ed1, _ed2 = st.columns([1, 2])
    with _ed1:
        _degree = st.selectbox("Minimum Degree", DEGREE_REQUIREMENTS, key="form_degree")
    with _ed2:
        _field = st.text_input(
            "Field of Study (optional)",
            placeholder="e.g. Computer Science, MBA",
            key="form_field",
        )

    _edu_c1, _edu_c2, _edu_c3 = st.columns(3)
    with _edu_c1:
        _tenth_min = st.slider("10th grade min %  (0 = any)", 0, 100, 0, key="form_tenth")
    with _edu_c2:
        _twelfth_min = st.slider("12th grade min %  (0 = any)", 0, 100, 0, key="form_twelfth")
    with _edu_c3:
        _cgpa_min = st.slider("Degree min CGPA /10  (0 = any)", 0.0, 10.0, 0.0, step=0.1, key="form_cgpa")

    st.markdown('<div class="section-heading">Experience &amp; Career Gap</div>', unsafe_allow_html=True)
    _gap_col1, _gap_col2 = st.columns([2, 1])
    with _gap_col1:
        _gap_max = st.slider(
            "Max acceptable career gap (months)  — 0 = no restriction",
            0, 36, 0, key="form_gap",
        )
    with _gap_col2:
        _relocate = st.checkbox("Must be willing to relocate", key="form_relocate")

    with st.expander("Inclusion Preferences (optional)", expanded=False):
        st.markdown(
            '<div class="info-box" style="margin-bottom:0.75rem;">'
            'These preferences are for targeted hiring programs. When set to <strong>No preference</strong>, '
            'the bias audit will automatically check for unintended demographic disparities.</div>',
            unsafe_allow_html=True,
        )
        _gender_pref = st.radio(
            "Gender preference", GENDER_OPTIONS, horizontal=True, key="form_gender_pref",
        )

    with st.expander("Preview generated JD text", expanded=False):
        _preview_form = {
            "profession": _profession,
            "seniority": _seniority,
            "experience_min": _exp_min,
            "experience_max": _exp_max,
            "work_mode": _work_mode,
            "skills": _selected_skills,
            "soft_skills": _soft_selected,
            "degree": _degree,
            "field_of_study": _field,
            "tenth_min": float(_tenth_min),
            "twelfth_min": float(_twelfth_min),
            "cgpa_min": float(_cgpa_min),
            "gap_max_months": _gap_max,
            "willing_to_relocate": _relocate,
            "gender_preference": _gender_pref,
        }
        _preview_text, _ = build_jd(_preview_form)
        st.code(_preview_text, language=None)

    _form_data = {
        "profession": _profession,
        "seniority": _seniority,
        "experience_min": _exp_min,
        "experience_max": _exp_max,
        "work_mode": _work_mode,
        "skills": _selected_skills,
        "soft_skills": _soft_selected,
        "degree": _degree,
        "field_of_study": _field,
        "tenth_min": float(_tenth_min),
        "twelfth_min": float(_twelfth_min),
        "cgpa_min": float(_cgpa_min),
        "gap_max_months": _gap_max,
        "willing_to_relocate": _relocate,
        "gender_preference": _gender_pref,
    }

# ── Paste / Upload tab ────────────────────────────────────────────────────────
_paste_jd_raw = ""
with jd_tab_text:
    _jd_upload_col, _jd_paste_col = st.columns([1, 1.6], gap="large")
    with _jd_upload_col:
        st.markdown(
            '<p style="font-size:0.82rem;font-weight:600;color:#475569;margin-bottom:0.4rem;">'
            'Upload a JD file (PDF or TXT)</p>',
            unsafe_allow_html=True,
        )
        _jd_file = st.file_uploader(
            "Upload JD", type=["pdf", "txt"], label_visibility="collapsed",
            key="jd_upload_file",
        )
        if _jd_file:
            import tempfile as _tf
            with _tf.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(_jd_file.name)[1]
            ) as _tmp:
                _tmp.write(_jd_file.read())
                _tmp_path = _tmp.name
            try:
                _paste_jd_raw = extract_text(_tmp_path)
                st.markdown(
                    f'<div class="success-box" style="margin-top:0.4rem;font-size:0.8rem;">'
                    f'&#10003; {_jd_file.name} loaded ({len(_paste_jd_raw):,} chars)</div>',
                    unsafe_allow_html=True,
                )
            except ValueError as e:
                st.markdown(f'<div class="error-box">{e}</div>', unsafe_allow_html=True)
            finally:
                os.unlink(_tmp_path)
    with _jd_paste_col:
        st.markdown(
            '<p style="font-size:0.82rem;font-weight:600;color:#475569;margin-bottom:0.4rem;">'
            'Or paste the job description text</p>',
            unsafe_allow_html=True,
        )
        _pasted = st.text_area(
            "Job description", height=220,
            placeholder="Paste the full job description here…",
            label_visibility="collapsed",
            key="jd_paste_area",
        )
        if not _paste_jd_raw and _pasted:
            _paste_jd_raw = _pasted

st.session_state["_form_data"] = _form_data
st.session_state["_paste_jd_raw"] = _paste_jd_raw

# ── Resume Files ──────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("### Resume Files")
res_tab_upload, res_tab_folder = st.tabs(["Upload files", "Load from folder"])

with res_tab_upload:
    uploaded = st.file_uploader(
        "Upload resumes", type=["pdf", "txt"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    if uploaded:
        st.markdown(
            f'<div class="info-box">{len(uploaded)} file(s) selected.</div>',
            unsafe_allow_html=True,
        )

with res_tab_folder:
    if st.button("Browse folder…", use_container_width=False):
        picked = _pick_folder()
        if picked:
            st.session_state["resume_folder"] = picked
    folder_path_str = st.session_state.get("resume_folder", "")
    if folder_path_str:
        st.markdown(
            f'<p style="font-size:0.8rem;color:#475569;margin:0.4rem 0 0.2rem;">'
            f'{folder_path_str}</p>',
            unsafe_allow_html=True,
        )
    folder_files: list[pathlib.Path] = []
    if folder_path_str.strip():
        folder_p = pathlib.Path(folder_path_str.strip())
        if not folder_p.exists():
            st.markdown('<div class="error-box">Folder not found.</div>', unsafe_allow_html=True)
        elif not folder_p.is_dir():
            st.markdown('<div class="error-box">Path is not a folder.</div>', unsafe_allow_html=True)
        else:
            folder_files = sorted(
                p for p in folder_p.iterdir()
                if p.suffix.lower() in {".pdf", ".txt"} and p.is_file()
            )
            if folder_files:
                st.markdown(
                    f'<div class="info-box">{len(folder_files)} file(s) found.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="warn-box">No .pdf or .txt files found in this folder.</div>',
                    unsafe_allow_html=True,
                )

resume_sources: list = folder_files if folder_files else (uploaded or [])

# ── Run Analysis ──────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
run = st.button("Run Analysis", type="primary")

if run:
    _active_form_data = st.session_state.get("_form_data", {})
    _active_paste_raw = st.session_state.get("_paste_jd_raw", "")
    _paste_has_content = bool(_active_paste_raw and _active_paste_raw.strip())

    if _active_form_data.get("skills"):
        jd_raw, _rubric = build_jd(_active_form_data)
    elif _paste_has_content:
        jd_raw = _active_paste_raw
        _rubric = None
    else:
        jd_raw = ""
        _rubric = None

    if not jd_raw or not jd_raw.strip():
        st.markdown(
            '<div class="warn-box">Please fill in the structured form (select at least one skill) '
            'or provide a job description in the Paste / Upload tab.</div>',
            unsafe_allow_html=True,
        )
    elif not resume_sources:
        st.markdown(
            '<div class="warn-box">Please upload resumes or select a folder before running.</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.status("Analysing resumes…", expanded=True) as _run_status:
            _pb = st.progress(0, text="Extracting resume text…")
            errors: list[str] = []
            candidates: list[Candidate] = []

            for i, f in enumerate(resume_sources):
                is_path = isinstance(f, pathlib.Path)
                fname = f.name
                suffix = pathlib.Path(fname).suffix.lower()

                if is_path:
                    tmp_path = str(f)
                    needs_cleanup = False
                else:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp.write(f.read())
                    tmp.close()
                    tmp_path = tmp.name
                    needs_cleanup = True

                try:
                    raw = extract_text(tmp_path)
                    candidates.append(Candidate(
                        filename=fname,
                        name=extract_name(raw),
                        raw_text=raw,
                        preprocessed_text=preprocess(raw),
                        matched_skills=extract_skills(raw),
                    ))
                except ValueError as e:
                    errors.append(f"{fname}: {e}")
                finally:
                    if needs_cleanup:
                        os.unlink(tmp_path)

                _pb.progress(
                    int((i + 1) / len(resume_sources) * 45),
                    text=f"Extracting text… ({i + 1}/{len(resume_sources)})",
                )

            for err in errors:
                st.warning(err)

            if candidates:
                _pb.progress(50, text="Scoring resumes…")
                jd_preprocessed = preprocess(jd_raw)
                jd_skills = extract_skills(jd_raw)
                raw_scores, norm_scores, _vectorizer, _tfidf_matrix = score_batch(
                    jd_preprocessed, [c.preprocessed_text for c in candidates],
                    return_vectorizer=True,
                )
                for c, raw_s, norm_s in zip(candidates, raw_scores, norm_scores):
                    c.raw_score = raw_s
                    c.score = norm_s
                    if _rubric:
                        c.rubric_result = apply_rubric(c.raw_text, _rubric)
                    apply_final_score(c)

                ranked = rank_candidates(candidates)
                _pb.progress(80, text="Running bias audit…")
                audit = run_audit(
                    ranked, jd_raw, jd_preprocessed,
                    gender_preference=(_rubric or {}).get("gender_preference", "No preference"),
                    gap_max_months=(_rubric or {}).get("max_gap_months", 0),
                )

                _pb.progress(100, text="Done!")
                st.session_state.update({
                    "ranked": ranked,
                    "audit": audit,
                    "jd_skills": jd_skills,
                    "rubric": _rubric,
                    "vectorizer": _vectorizer,
                    "tfidf_matrix": _tfidf_matrix,
                    "tfidf_order": [c.filename for c in candidates],
                    "ready": True,
                })