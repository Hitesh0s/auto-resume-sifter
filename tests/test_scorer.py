import pytest
from src.scorer import score_batch


def test_identical_document_scores_100():
    text = "python django flask postgresql docker kubernetes"
    _, norm = score_batch(text, [text])
    assert norm[0] == pytest.approx(100.0, abs=0.1)


def test_empty_resume_list_returns_empty():
    raw, norm = score_batch("some job description", [])
    assert raw == [] and norm == []


def test_scores_are_between_0_and_100():
    jd = "python machine learning tensorflow scikit-learn"
    resumes = [
        "java spring boot microservices",
        "python tensorflow keras deep learning",
        "marketing sales customer service",
    ]
    raw, norm = score_batch(jd, resumes)
    assert len(raw) == 3 and len(norm) == 3
    for s in raw + norm:
        assert 0.0 <= s <= 100.0


def test_relevant_resume_scores_higher_than_irrelevant():
    jd = "python django postgresql docker"
    relevant = "python django flask postgresql rest api docker"
    irrelevant = "marketing brand strategy social media content"
    raw, norm = score_batch(jd, [relevant, irrelevant])
    assert raw[0] > raw[1]
    assert norm[0] > norm[1]


def test_single_word_jd_does_not_crash():
    raw, norm = score_batch("python", ["python developer", "java developer"])
    assert len(raw) == 2 and len(norm) == 2
