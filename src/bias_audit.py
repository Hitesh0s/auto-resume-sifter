"""
Multi-dimensional demographic bias detection and correction.

Dimensions audited:
  1. Gender        — inferred from first name (lookup) + pronoun scan fallback
  2. Career Gap    — detect if gapped candidates score lower (advisory only)
  3. Institution   — top-tier vs other-tier institution prestige (advisory only)
  4. Socio-cultural — India-specific surname proxy: region, religion, caste group

Statistical tests:
  - Mann-Whitney U (2 groups) or Kruskal-Wallis (3+ groups) from scipy.stats
  - Disparate Impact Ratio (DIR): lower_group_mean / higher_group_mean
    DIR < 0.8 = 4/5ths rule violation (US EEOC guideline)
  - pp-gap: |mean_A - mean_B| > BIAS_THRESHOLD (10 pp)

Corrections:
  - Gender: name-masking counterfactual (all occurrences), up to 3 iterative rounds
  - All other dimensions: advisory only (signals may be intentional JD criteria)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import scipy.stats as _stats

from src.scorer import score_batch
from src.preprocessor import preprocess

BIAS_THRESHOLD = 10.0      # percentage points
DIR_THRESHOLD = 0.80       # 4/5ths rule
LOW_SAMPLE_THRESHOLD = 5

# ── Gender lookup ──────────────────────────────────────────────────────────────
_GENDER_MAP: dict[str, str] = {
    # Male
    "james": "Male", "john": "Male", "robert": "Male", "michael": "Male",
    "william": "Male", "david": "Male", "richard": "Male", "joseph": "Male",
    "thomas": "Male", "charles": "Male", "daniel": "Male", "matthew": "Male",
    "anthony": "Male", "mark": "Male", "donald": "Male", "steven": "Male",
    "paul": "Male", "andrew": "Male", "ryan": "Male", "kevin": "Male",
    "rahul": "Male", "arjun": "Male", "rohan": "Male", "amit": "Male",
    "suresh": "Male", "vijay": "Male", "ravi": "Male", "kiran": "Male",
    "sanjay": "Male", "nikhil": "Male", "hitesh": "Male", "rishi": "Male",
    "aarav": "Male", "aditya": "Male", "vishal": "Male", "vikram": "Male",
    "rajesh": "Male", "mukesh": "Male", "naresh": "Male", "sunil": "Male",
    "pranav": "Male", "dhruv": "Male", "gaurav": "Male", "varun": "Male",
    "harsh": "Male", "shubham": "Male", "siddharth": "Male", "ankit": "Male",
    # Female
    "mary": "Female", "patricia": "Female", "jennifer": "Female", "linda": "Female",
    "barbara": "Female", "elizabeth": "Female", "susan": "Female", "jessica": "Female",
    "sarah": "Female", "karen": "Female", "lisa": "Female", "nancy": "Female",
    "betty": "Female", "helen": "Female", "sandra": "Female", "ashley": "Female",
    "emily": "Female", "donna": "Female", "michelle": "Female", "carol": "Female",
    "priya": "Female", "anjali": "Female", "pooja": "Female", "neha": "Female",
    "divya": "Female", "kavya": "Female", "sneha": "Female", "ananya": "Female",
    "shreya": "Female", "meera": "Female", "ishita": "Female", "riya": "Female",
    "kritika": "Female", "aishwarya": "Female", "simran": "Female", "nisha": "Female",
    "radhika": "Female", "shalini": "Female", "swati": "Female", "deepika": "Female",
    "lakshmi": "Female", "gayatri": "Female", "bhavana": "Female", "madhuri": "Female",
}

_MALE_PRONOUNS = re.compile(r"\b(he|him|his)\b", re.IGNORECASE)
_FEMALE_PRONOUNS = re.compile(r"\b(she|her|hers)\b", re.IGNORECASE)


def _pronouns_gender(raw_text: str) -> str | None:
    mc = len(_MALE_PRONOUNS.findall(raw_text))
    fc = len(_FEMALE_PRONOUNS.findall(raw_text))
    if mc >= 2 and mc > fc * 1.5:
        return "Male"
    if fc >= 2 and fc > mc * 1.5:
        return "Female"
    return None


def infer_gender(name: str | None, raw_text: str | None = None) -> str:
    if name:
        first = name.strip().split()[0].lower()
        if first in _GENDER_MAP:
            return _GENDER_MAP[first]
    if raw_text:
        pg = _pronouns_gender(raw_text)
        if pg:
            return pg
    return "Unknown"


def _infer_method(name: str | None, raw_text: str | None) -> str:
    if name:
        first = name.strip().split()[0].lower()
        if first in _GENDER_MAP:
            return "name lookup"
    if raw_text and _pronouns_gender(raw_text):
        return "pronoun scan"
    return "not determined"


def mask_name(text: str, name: str) -> str:
    """Replace all occurrences of a candidate's name with [CANDIDATE]."""
    escaped = re.escape(name)
    return re.sub(escaped, "[CANDIDATE]", text, flags=re.IGNORECASE)


# ── Statistical helpers ────────────────────────────────────────────────────────

def _compute_group_stats(group_scores: dict[str, list[float]]) -> dict[str, dict]:
    result = {}
    for g, scores in group_scores.items():
        if not scores:
            continue
        result[g] = {
            "count": len(scores),
            "mean_score": round(sum(scores) / len(scores), 1),
        }
    return result


def _disparity_test(
    group_stats: dict[str, dict],
    group_scores: dict[str, list[float]],
    exclude_unknown: bool = True,
) -> dict:
    """
    Returns {bias_detected, pp_gap, dir, p_value, test_name, low_sample_groups}.
    """
    stats_subset = {
        g: s for g, s in group_stats.items()
        if (not exclude_unknown or g != "Unknown") and s["count"] >= 3
    }
    low_sample = [g for g, s in group_stats.items() if s["count"] < LOW_SAMPLE_THRESHOLD]
    known = {g: s for g, s in stats_subset.items()}
    if len(known) < 2:
        return {
            "bias_detected": False, "pp_gap": 0.0, "dir": 1.0,
            "p_value": None, "test_name": "—", "low_sample_groups": low_sample,
            "insufficient_data": True,
        }

    means = {g: s["mean_score"] for g, s in known.items()}
    pp_gap = round(max(means.values()) - min(means.values()), 1)
    lower_mean = min(means.values())
    higher_mean = max(means.values())
    dir_val = round(lower_mean / higher_mean, 3) if higher_mean else 1.0

    # Statistical test
    groups_list = [group_scores[g] for g in known if g in group_scores and len(group_scores[g]) >= 3]
    p_value = None
    test_name = "—"
    if len(groups_list) == 2:
        try:
            _, p_value = _stats.mannwhitneyu(*groups_list, alternative="two-sided")
            p_value = round(p_value, 4)
            test_name = "Mann-Whitney U"
        except Exception:
            pass
    elif len(groups_list) > 2:
        try:
            _, p_value = _stats.kruskal(*groups_list)
            p_value = round(p_value, 4)
            test_name = "Kruskal-Wallis"
        except Exception:
            pass

    bias_detected = (pp_gap > BIAS_THRESHOLD) or (dir_val < DIR_THRESHOLD)
    if p_value is not None and p_value >= 0.05:
        bias_detected = False  # not statistically significant

    return {
        "bias_detected": bias_detected,
        "pp_gap": pp_gap,
        "dir": dir_val,
        "p_value": p_value,
        "test_name": test_name,
        "low_sample_groups": low_sample,
        "insufficient_data": False,
        "group_means": means,
        "lower_group": min(means, key=means.get),
        "higher_group": max(means, key=means.get),
    }


# ── AuditResult dataclass ─────────────────────────────────────────────────────

@dataclass
class DimensionResult:
    """Result for a single bias audit dimension."""
    label: str                         # "Gender", "Career Gap", etc.
    audited: bool                      # False = skipped (intentional criteria)
    skip_reason: str = ""
    bias_detected: bool = False
    advisory_only: bool = False        # True = no auto-correction
    corrected: bool = False
    pp_gap: float = 0.0
    dir_value: float = 1.0
    p_value: float | None = None
    test_name: str = "—"
    group_stats: dict = field(default_factory=dict)
    group_scores: dict = field(default_factory=dict)
    low_sample_groups: list = field(default_factory=list)
    insufficient_data: bool = False
    candidate_deltas: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class AuditResult:
    bias_detected: bool                 # True if ANY correctable dimension flagged
    group_stats: dict[str, dict]        # Gender group stats (backward compat)
    corrected: bool
    candidate_deltas: list[dict]
    low_sample_groups: list[str] = field(default_factory=list)
    inference_rows: list[dict] = field(default_factory=list)
    gender_audit_skipped: bool = False
    gender_audit_skip_reason: str = ""
    gap_audit_skipped: bool = False
    gap_audit_skip_reason: str = ""
    dimension_results: dict[str, DimensionResult] = field(default_factory=dict)


# ── Counterfactual re-scoring ─────────────────────────────────────────────────

def _counterfactual_correction(
    candidates,
    jd_preprocessed: str,
    score_attr: str = "final_score",
) -> list[dict]:
    """
    Mask all name occurrences, re-score, average original + masked.
    Up to 3 iterative rounds until max delta < 2 pp.
    Returns list of candidate delta dicts.
    """
    from src.classifier import apply_final_score, classify

    deltas: list[dict] = []
    for _round in range(3):
        masked_texts = [
            preprocess(mask_name(c.raw_text, c.name)) if c.name else preprocess(c.raw_text)
            for c in candidates
        ]
        _, masked_norm = score_batch(jd_preprocessed, masked_texts)

        round_deltas: list[dict] = []
        max_delta = 0.0
        for c, masked_s in zip(candidates, masked_norm):
            orig = getattr(c, score_attr, None) or c.score
            corrected = round((orig + masked_s) / 2, 1)
            delta = abs(corrected - orig)
            max_delta = max(max_delta, delta)
            round_deltas.append({
                "filename": c.filename,
                "name": c.name or "Unknown",
                "original_score": orig,
                "masked_score": round(masked_s, 1),
                "corrected_score": corrected,
            })
            c.final_score = corrected
            c.score = corrected
            c.tier = classify(corrected)

        deltas = round_deltas
        if max_delta < 2.0:
            break

    return deltas


# ── Main audit function ────────────────────────────────────────────────────────

def run_audit(
    candidates,
    jd_text: str,
    jd_preprocessed: str,
    gender_preference: str = "No preference",
    gap_max_months: int = 0,
) -> AuditResult:
    """Run all bias audit dimensions on a ranked candidate list."""

    from src.gap_parser import enrich_candidate

    # Enrich all candidates with gap/institution/proxy data
    for c in candidates:
        enrich_candidate(c)

    # ── Gender inference ──────────────────────────────────────────────────────
    inference_rows: list[dict] = []
    for c in candidates:
        gender = infer_gender(c.name, c.raw_text)
        c._gender = gender  # type: ignore[attr-defined]
        method = _infer_method(c.name, c.raw_text)
        inference_rows.append({
            "filename": c.filename,
            "name": c.name or "—",
            "inferred_gender": gender,
            "method": method,
        })

    _score_of = lambda c: (getattr(c, "final_score", None) or c.score)

    # ── Dimension 1: Gender ───────────────────────────────────────────────────
    gender_skipped = bool(gender_preference and gender_preference != "No preference")
    gender_skip_reason = (
        f"Gender preference explicitly set to '{gender_preference}' in the JD form."
    ) if gender_skipped else ""

    gender_group_scores: dict[str, list[float]] = {}
    for c in candidates:
        gender_group_scores.setdefault(c._gender, []).append(_score_of(c))  # type: ignore

    gender_group_stats = _compute_group_stats(gender_group_scores)
    gender_dim = DimensionResult(
        label="Gender",
        audited=not gender_skipped,
        skip_reason=gender_skip_reason,
        group_stats=gender_group_stats,
        group_scores=gender_group_scores,
        low_sample_groups=[g for g, s in gender_group_stats.items() if s["count"] < LOW_SAMPLE_THRESHOLD],
    )

    candidate_deltas: list[dict] = []
    overall_bias = False
    overall_corrected = False

    if not gender_skipped:
        test = _disparity_test(gender_group_stats, gender_group_scores)
        gender_dim.bias_detected = test["bias_detected"]
        gender_dim.pp_gap = test["pp_gap"]
        gender_dim.dir_value = test["dir"]
        gender_dim.p_value = test["p_value"]
        gender_dim.test_name = test["test_name"]
        gender_dim.low_sample_groups = test["low_sample_groups"]
        gender_dim.insufficient_data = test.get("insufficient_data", False)
        gender_dim.extra = {k: v for k, v in test.items()
                            if k not in ("bias_detected", "pp_gap", "dir", "p_value",
                                         "test_name", "low_sample_groups", "insufficient_data")}
        if gender_dim.bias_detected:
            candidate_deltas = _counterfactual_correction(candidates, jd_preprocessed)
            gender_dim.candidate_deltas = candidate_deltas
            gender_dim.corrected = True
            overall_bias = True
            overall_corrected = True

    # ── Dimension 2: Career Gap ───────────────────────────────────────────────
    gap_skipped = gap_max_months > 0
    gap_skip_reason = (
        f"Career gap max of {gap_max_months} months was set in the JD — gap scoring is intentional."
    ) if gap_skipped else ""

    gap_group_scores: dict[str, list[float]] = {}
    for c in candidates:
        has_gap = "Has Gap (≥6m)" if c.career_gap_months >= 6 else "No Significant Gap"
        gap_group_scores.setdefault(has_gap, []).append(_score_of(c))

    gap_group_stats = _compute_group_stats(gap_group_scores)
    gap_dim = DimensionResult(
        label="Career Gap",
        audited=not gap_skipped,
        skip_reason=gap_skip_reason,
        advisory_only=True,
        group_stats=gap_group_stats,
        group_scores=gap_group_scores,
        low_sample_groups=[g for g, s in gap_group_stats.items() if s["count"] < LOW_SAMPLE_THRESHOLD],
    )

    if not gap_skipped and not gap_dim.insufficient_data:
        gap_test = _disparity_test(gap_group_stats, gap_group_scores, exclude_unknown=False)
        gap_dim.bias_detected = gap_test["bias_detected"]
        gap_dim.pp_gap = gap_test["pp_gap"]
        gap_dim.dir_value = gap_test["dir"]
        gap_dim.p_value = gap_test["p_value"]
        gap_dim.test_name = gap_test["test_name"]
        gap_dim.insufficient_data = gap_test.get("insufficient_data", False)

    # ── Dimension 3: Institution Tier ─────────────────────────────────────────
    inst_group_scores: dict[str, list[float]] = {}
    for c in candidates:
        if c.institution_tier != "Unknown":
            inst_group_scores.setdefault(c.institution_tier, []).append(_score_of(c))

    inst_group_stats = _compute_group_stats(inst_group_scores)
    inst_dim = DimensionResult(
        label="Institution Tier",
        audited=True,
        advisory_only=True,
        group_stats=inst_group_stats,
        group_scores=inst_group_scores,
        low_sample_groups=[g for g, s in inst_group_stats.items() if s["count"] < LOW_SAMPLE_THRESHOLD],
    )

    if inst_group_stats:
        inst_test = _disparity_test(inst_group_stats, inst_group_scores, exclude_unknown=False)
        inst_dim.bias_detected = inst_test["bias_detected"]
        inst_dim.pp_gap = inst_test["pp_gap"]
        inst_dim.dir_value = inst_test["dir"]
        inst_dim.p_value = inst_test["p_value"]
        inst_dim.test_name = inst_test["test_name"]
        inst_dim.insufficient_data = inst_test.get("insufficient_data", False)

    # ── Dimension 4: Socio-cultural proxy ─────────────────────────────────────
    # Run three sub-audits: region, religion, caste_group
    socio_dims: dict[str, DimensionResult] = {}
    for _attr, _label in [
        ("region_proxy", "Region"),
        ("religion_proxy", "Religion"),
        ("caste_group_proxy", "Caste Group"),
    ]:
        _sub_scores: dict[str, list[float]] = {}
        for c in candidates:
            val = getattr(c, _attr, "Unknown")
            if val != "Unknown":
                _sub_scores.setdefault(val, []).append(_score_of(c))
        _sub_stats = _compute_group_stats(_sub_scores)
        _sub_dim = DimensionResult(
            label=_label,
            audited=True,
            advisory_only=False,  # name-masking correction applies
            group_stats=_sub_stats,
            group_scores=_sub_scores,
            low_sample_groups=[g for g, s in _sub_stats.items() if s["count"] < LOW_SAMPLE_THRESHOLD],
        )
        if _sub_stats:
            _test = _disparity_test(_sub_stats, _sub_scores, exclude_unknown=False)
            _sub_dim.bias_detected = _test["bias_detected"]
            _sub_dim.pp_gap = _test["pp_gap"]
            _sub_dim.dir_value = _test["dir"]
            _sub_dim.p_value = _test["p_value"]
            _sub_dim.test_name = _test["test_name"]
            _sub_dim.insufficient_data = _test.get("insufficient_data", False)
            if _sub_dim.bias_detected and not _test.get("insufficient_data"):
                # Apply name-masking correction for detectable socio-cultural bias
                _deltas = _counterfactual_correction(candidates, jd_preprocessed)
                _sub_dim.candidate_deltas = _deltas
                _sub_dim.corrected = True
                overall_bias = True
                overall_corrected = True

        socio_dims[_attr] = _sub_dim

    socio_combined_dim = DimensionResult(
        label="Socio-cultural Proxy (India)",
        audited=True,
        advisory_only=False,
        bias_detected=any(d.bias_detected for d in socio_dims.values()),
        extra={"sub_dimensions": socio_dims},
    )

    return AuditResult(
        bias_detected=overall_bias,
        group_stats=gender_group_stats,
        corrected=overall_corrected,
        candidate_deltas=candidate_deltas,
        low_sample_groups=gender_dim.low_sample_groups,
        inference_rows=inference_rows,
        gender_audit_skipped=gender_skipped,
        gender_audit_skip_reason=gender_skip_reason,
        gap_audit_skipped=gap_skipped,
        gap_audit_skip_reason=gap_skip_reason,
        dimension_results={
            "gender": gender_dim,
            "career_gap": gap_dim,
            "institution_tier": inst_dim,
            "socio_cultural": socio_combined_dim,
        },
    )
