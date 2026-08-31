from src.classifier import classify, rank_candidates, Candidate, TIER_STRONG, TIER_PARTIAL, TIER_UNSUITABLE


# ── classify() boundary tests ─────────────────────────────────────────────────

def test_strong_match_exactly_70():
    assert classify(70.0) == TIER_STRONG


def test_strong_match_above_70():
    assert classify(85.5) == TIER_STRONG


def test_partial_match_exactly_40():
    assert classify(40.0) == TIER_PARTIAL


def test_partial_match_just_below_70():
    assert classify(69.9) == TIER_PARTIAL


def test_not_suitable_just_below_40():
    assert classify(39.9) == TIER_UNSUITABLE


def test_not_suitable_zero():
    assert classify(0.0) == TIER_UNSUITABLE


def test_not_suitable_exactly_0():
    assert classify(0.0) == TIER_UNSUITABLE


# ── rank_candidates() tests ───────────────────────────────────────────────────

def _make(score, tier):
    c = Candidate(filename="test.pdf", name=None, raw_text="", preprocessed_text="")
    c.score = score
    c.tier = tier
    return c


def test_strong_comes_before_partial():
    candidates = [
        _make(55.0, TIER_PARTIAL),
        _make(75.0, TIER_STRONG),
    ]
    ranked = rank_candidates(candidates)
    assert ranked[0].tier == TIER_STRONG


def test_partial_comes_before_unsuitable():
    candidates = [
        _make(20.0, TIER_UNSUITABLE),
        _make(50.0, TIER_PARTIAL),
    ]
    ranked = rank_candidates(candidates)
    assert ranked[0].tier == TIER_PARTIAL


def test_within_tier_sorted_by_score_desc():
    candidates = [
        _make(72.0, TIER_STRONG),
        _make(90.0, TIER_STRONG),
        _make(80.0, TIER_STRONG),
    ]
    ranked = rank_candidates(candidates)
    scores = [c.score for c in ranked]
    assert scores == sorted(scores, reverse=True)


def test_empty_list_returns_empty():
    assert rank_candidates([]) == []
