from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def score_batch(
    jd_text: str,
    resume_texts: list[str],
    return_vectorizer: bool = False,
) -> tuple:
    """Score each resume against the JD using TF-IDF cosine similarity.

    Parameters
    ----------
    jd_text : str
        Preprocessed job description text.
    resume_texts : list[str]
        Preprocessed resume texts.
    return_vectorizer : bool
        If True, return (raw, normalised, vectorizer, tfidf_matrix) instead
        of (raw, normalised). Used by the visualisation layer.

    Returns
    -------
    (raw_scores, normalised_scores) — always present.
    (raw_scores, normalised_scores, vectorizer, tfidf_matrix) — when return_vectorizer=True.
    """
    if not resume_texts:
        if return_vectorizer:
            return [], [], None, None
        return [], []

    corpus = [jd_text] + resume_texts
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=1, ngram_range=(1, 1))
    matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(matrix[0], matrix[1:])[0]

    raw = [round(float(s) * 100, 1) for s in similarities]

    top = max(raw) if raw else 0.0
    normalised = [round(s / top * 100, 1) for s in raw] if top > 0 else raw[:]

    if return_vectorizer:
        return raw, normalised, vectorizer, matrix
    return raw, normalised


def top_tfidf_terms(
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    doc_index: int,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Return the top-N TF-IDF weighted terms for a document in the matrix.

    doc_index 0 is the JD; resume indices start at 1.
    Returns list of (term, weight) sorted descending by weight.
    """
    feature_names = vectorizer.get_feature_names_out()
    row = np.asarray(tfidf_matrix[doc_index].todense()).flatten()
    top_indices = row.argsort()[::-1][:top_n]
    return [(feature_names[i], round(float(row[i]), 4)) for i in top_indices if row[i] > 0]
