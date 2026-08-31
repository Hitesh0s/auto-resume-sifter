"""
Applies a structured rubric against a candidate's raw text to produce a
RubricResult: knockout flag, bonus multiplier, and human-readable reasons.

The rubric is produced by jd_builder.build_jd() from a filled JD form.
Final score blend (in classifier.py): tfidf_normalised * 0.70 + rubric_bonus * 0.30
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RubricResult:
    knockout: bool = False          # True → candidate is Not Suitable regardless of TF-IDF
    bonus: float = 0.0              # 0.0–1.0 additive contribution to final score
    knockout_reasons: list[str] = field(default_factory=list)
    bonus_reasons: list[str] = field(default_factory=list)


# ── Grade / CGPA extraction helpers ──────────────────────────────────────────

_TENTH_PATTERNS = [
    r"10th[^\d]{0,15}(\d{2,3}\.?\d*)\s*%",
    r"sslc[^\d]{0,15}(\d{2,3}\.?\d*)\s*%",
    r"class\s*x[^\d]{0,15}(\d{2,3}\.?\d*)\s*%",
    r"matriculation[^\d]{0,15}(\d{2,3}\.?\d*)\s*%",
]
_TWELFTH_PATTERNS = [
    r"12th[^\d]{0,15}(\d{2,3}\.?\d*)\s*%",
    r"hsc[^\d]{0,15}(\d{2,3}\.?\d*)\s*%",
    r"class\s*xii[^\d]{0,15}(\d{2,3}\.?\d*)\s*%",
    r"intermediate[^\d]{0,15}(\d{2,3}\.?\d*)\s*%",
    r"higher\s*secondary[^\d]{0,15}(\d{2,3}\.?\d*)\s*%",
]
_CGPA_PATTERNS = [
    r"cgpa[^\d]{0,10}(\d\.\d{1,2})",
    r"gpa[^\d]{0,10}(\d\.\d{1,2})",
    r"(\d\.\d{1,2})\s*/\s*10",
    r"(\d\.\d{1,2})\s*/\s*4",  # US-style, treated as ×2.5
]

_DEGREE_KEYWORDS = {
    "Diploma": ["diploma", "polytechnic"],
    "Bachelor's": ["bachelor", "b.tech", "b.e.", "b.sc", "bca", "bba", "b.com", "undergraduate"],
    "Master's": ["master", "m.tech", "m.e.", "m.sc", "mca", "mba", "postgraduate"],
    "PhD": ["ph.d", "phd", "doctorate", "doctor of"],
}


def _extract_float(text: str, patterns: list[str]) -> float | None:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def _has_degree(text: str, degree: str) -> bool:
    if not degree or degree == "Any":
        return True
    keywords = _DEGREE_KEYWORDS.get(degree, [])
    t = text.lower()
    return any(kw in t for kw in keywords)


# ── Skill matching ────────────────────────────────────────────────────────────

def _count_skill_matches(text: str, skills: list[str]) -> int:
    t = text.lower()
    return sum(1 for s in skills if re.search(r"\b" + re.escape(s.lower()) + r"\b", t))


# ── Main entry point ──────────────────────────────────────────────────────────

def apply_rubric(raw_text: str, rubric: dict) -> RubricResult:
    """
    Evaluate a candidate's raw resume text against the structured rubric.

    Parameters
    ----------
    raw_text : str
        Unprocessed resume text (from reader.extract_text).
    rubric : dict
        Produced by jd_builder.build_jd(); keys: required_skills,
        education_floor, max_gap_months, gender_preference, skill_weights.

    Returns
    -------
    RubricResult
    """
    result = RubricResult()
    edu_floor: dict = rubric.get("education_floor", {})
    required_skills: list[str] = rubric.get("required_skills", [])
    skill_weights: dict = rubric.get("skill_weights", {})

    # ── Education degree knockout ─────────────────────────────────────────────
    degree_req = edu_floor.get("degree", "Any")
    if degree_req and degree_req != "Any":
        if not _has_degree(raw_text, degree_req):
            result.knockout = True
            result.knockout_reasons.append(
                f"Degree requirement not met: {degree_req} not detected in resume."
            )

    # ── Education score knockouts ─────────────────────────────────────────────
    tenth_min = edu_floor.get("tenth_min", 0.0)
    if tenth_min:
        tenth_score = _extract_float(raw_text, _TENTH_PATTERNS)
        if tenth_score is not None and tenth_score < tenth_min:
            result.knockout = True
            result.knockout_reasons.append(
                f"10th grade score {tenth_score:.1f}% is below the required {tenth_min:.0f}%."
            )

    twelfth_min = edu_floor.get("twelfth_min", 0.0)
    if twelfth_min:
        twelfth_score = _extract_float(raw_text, _TWELFTH_PATTERNS)
        if twelfth_score is not None and twelfth_score < twelfth_min:
            result.knockout = True
            result.knockout_reasons.append(
                f"12th grade score {twelfth_score:.1f}% is below the required {twelfth_min:.0f}%."
            )

    cgpa_min = edu_floor.get("cgpa_min", 0.0)
    if cgpa_min:
        cgpa = _extract_float(raw_text, _CGPA_PATTERNS)
        if cgpa is not None:
            # Normalise US 4.0 scale to 10.0
            if cgpa <= 4.5:
                cgpa = cgpa * 2.5
            if cgpa < cgpa_min:
                result.knockout = True
                result.knockout_reasons.append(
                    f"Degree CGPA {cgpa:.2f}/10 is below the required {cgpa_min:.1f}/10."
                )

    # ── Skill coverage bonus ──────────────────────────────────────────────────
    if required_skills:
        matched = _count_skill_matches(raw_text, required_skills)
        coverage = matched / len(required_skills)
        # Bonus: 0 at 0% coverage, 1.0 at 100% coverage (linear)
        result.bonus = round(coverage, 4)
        result.bonus_reasons.append(
            f"{matched}/{len(required_skills)} required skills found "
            f"({coverage * 100:.0f}% coverage)."
        )

    return result
