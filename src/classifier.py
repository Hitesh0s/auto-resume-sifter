from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rubric_scorer import RubricResult

STRONG_THRESHOLD = 70.0
PARTIAL_THRESHOLD = 40.0

TIER_STRONG = "Strong Match"
TIER_PARTIAL = "Partial Match"
TIER_UNSUITABLE = "Not Suitable"

_TIER_ORDER = {TIER_STRONG: 0, TIER_PARTIAL: 1, TIER_UNSUITABLE: 2}

# Weight blend: TF-IDF normalised (70%) + rubric skill-coverage bonus (30%)
_TFIDF_WEIGHT = 0.70
_RUBRIC_WEIGHT = 0.30


@dataclass
class Candidate:
    filename: str
    name: str | None
    raw_text: str
    preprocessed_text: str
    raw_score: float = 0.0          # TF-IDF cosine × 100, un-normalised
    score: float = 0.0              # normalised TF-IDF score (best = 100)
    final_score: float = 0.0       # blended score used for tier + ranking
    tier: str = ""
    matched_skills: list[str] = field(default_factory=list)
    rubric_result: object = None    # RubricResult | None
    career_gap_months: int = 0
    institution_tier: str = "Unknown"
    region_proxy: str = "Unknown"
    religion_proxy: str = "Unknown"
    caste_group_proxy: str = "Unknown"
    tfidf_row: int = -1


def classify(score: float) -> str:
    if score >= STRONG_THRESHOLD:
        return TIER_STRONG
    if score >= PARTIAL_THRESHOLD:
        return TIER_PARTIAL
    return TIER_UNSUITABLE


def apply_final_score(candidate: Candidate) -> None:
    """Compute final_score from TF-IDF score and rubric bonus, then set tier.

    If rubric has a knockout, tier is forced to Not Suitable regardless of TF-IDF.
    """
    rr = candidate.rubric_result
    if rr is not None and rr.knockout:
        candidate.final_score = 0.0
        candidate.tier = TIER_UNSUITABLE
        return

    if rr is not None:
        bonus_pct = rr.bonus * 100
        blended = candidate.score * _TFIDF_WEIGHT + bonus_pct * _RUBRIC_WEIGHT
    else:
        blended = candidate.score  # no rubric — use normalised TF-IDF directly

    candidate.final_score = round(min(blended, 100.0), 1)
    candidate.tier = classify(candidate.final_score)


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    sort_score = lambda c: c.final_score if c.final_score > 0 else c.score
    return sorted(
        candidates,
        key=lambda c: (_TIER_ORDER.get(c.tier, 3), -sort_score(c)),
    )
