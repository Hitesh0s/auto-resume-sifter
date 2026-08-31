# Auto Resume Sifter

An end-to-end AI-assisted resume screening and ranking system built with Python and Streamlit. Designed as a university final-year project, it combines classical NLP (TF-IDF cosine similarity) with structured rubric scoring, multi-dimensional bias detection, and a production-quality multi-page web interface.

---

## Table of Contents

1. [Overview](#overview)
2. [Feature Summary](#feature-summary)
3. [System Architecture](#system-architecture)
4. [Algorithms & Technical Detail](#algorithms--technical-detail)
   - [Text Preprocessing](#1-text-preprocessing)
   - [TF-IDF Vectorisation & Cosine Similarity](#2-tf-idf-vectorisation--cosine-similarity)
   - [Score Normalisation](#3-score-normalisation)
   - [Rubric Scoring](#4-rubric-scoring)
   - [Tier Classification](#5-tier-classification)
   - [Skill & Name Extraction](#6-skill--name-extraction)
   - [Bias Audit](#7-bias-audit)
5. [Bias Audit Deep-Dive](#bias-audit-deep-dive)
6. [UI & Navigation](#ui--navigation)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Project Structure](#project-structure)
10. [Testing](#testing)
11. [Limitations](#limitations)
12. [Dependencies](#dependencies)
13. [Data Source](#data-source)

---

## Overview

Auto Resume Sifter addresses a well-documented problem in HR technology: manual resume screening is slow, inconsistent, and susceptible to unconscious bias. This system automates ranking using two complementary signals — content similarity (TF-IDF cosine) and structured criteria matching (rubric) — and then audits those rankings for demographic disparities across four dimensions.

The project was built entirely in Python using open-source libraries, with no external paid API calls. All scoring and auditing is transparent and explainable.

---

## Feature Summary

| Feature | Description |
|---|---|
| **Structured JD Builder** | Form-based job description builder with profession/seniority/skills dropdowns; generates a structured JD text and a weighted rubric automatically |
| **Paste / Upload JD** | Alternative input: paste plain text or upload a PDF/TXT file |
| **Multi-mode resume input** | Upload individual files (PDF/TXT) or select a folder with a native OS picker |
| **TF-IDF cosine scoring** | Corpus-level TF-IDF vectorisation; cosine similarity of each resume against the JD |
| **Score normalisation** | Batch-relative ranking so the top candidate is always 100%; fixed thresholds remain meaningful regardless of corpus |
| **Rubric scoring** | Weighted structured criteria (skills, education, experience, career gap, relocation); blended 70/30 with TF-IDF when a structured JD is used |
| **Tier classification** | Three tiers: Strong Match (≥ 70%), Partial Match (≥ 40%), Not Suitable (< 40%) |
| **Gap analysis** | Per-candidate expanders for Partial matches showing matched/missing skills and a rule-based recommendation |
| **Score analytics** | Interactive Plotly charts: score distribution histogram, top TF-IDF discriminating terms, skill coverage heatmap |
| **Gender bias audit** | First-name lookup + pronoun-scan inference; Mann-Whitney U test; counterfactual name-masking correction |
| **Career gap audit** | Detects systematic score penalty for candidates with employment gaps |
| **Institution tier audit** | Checks whether graduates of top-tier institutions are being over-favoured |
| **Socio-cultural audit (India)** | Surname-based proxy for region, religion, and caste group; statistical disparity detection; constitutional context for caste findings |
| **Inference transparency** | Per-candidate table showing how gender was inferred (name lookup vs pronoun scan) |
| **Authentication** | Login-gated access with bcrypt-hashed credentials stored in YAML; session cookie management via streamlit-authenticator |
| **Demo data** | 15 HuggingFace resumes rendered as formatted PDFs (5 × Software Engineer, Data Scientist, HR Specialist) |
| **CSV export** | Download ranked results with raw score, normalised score, tier, and matched skills |
| **Multi-page navigation** | Persistent top-bar navigation (Upload & Analyse / Results / Bias Audit) with session-state preserved across pages |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       app.py (entry point)                  │
│  Auth gate → CSS injection → top navbar → st.navigation()  │
└────────────┬────────────────────────────────────────────────┘
             │ pg.run()
    ┌────────▼────────┐  ┌──────────────┐  ┌──────────────┐
    │ pages/upload.py │  │pages/results │  │ pages/bias.py│
    │ JD builder      │  │ Stat cards   │  │ 4-step audit │
    │ Resume upload   │  │ Results table│  │ Disparity    │
    │ Pipeline runner │  │ Gap analysis │  │ charts       │
    └────────┬────────┘  └──────────────┘  └──────────────┘
             │
    ┌────────▼──────────────────────────────────────────────┐
    │                   src/ modules                         │
    │                                                        │
    │  reader.py        →  PDF + TXT text extraction        │
    │  preprocessor.py  →  NLP cleaning pipeline            │
    │  extractor.py     →  spaCy NER + skill keywords       │
    │  scorer.py        →  TF-IDF + cosine similarity       │
    │  classifier.py    →  Tier logic + Candidate dataclass │
    │  rubric_scorer.py →  Weighted structured criteria     │
    │  bias_audit.py    →  Multi-dim fairness analysis      │
    │  jd_builder.py    →  JD text generator + rubric       │
    │  ui_helpers.py    →  Shared CSS + formatting funcs    │
    └───────────────────────────────────────────────────────┘
```

**Session state** is used to pass data between pages without re-running the pipeline. After analysis, `st.session_state` holds: `ranked` (list of Candidate objects), `audit` (AuditResult), `jd_skills`, `rubric`, `vectorizer`, and `tfidf_matrix`. The Results and Bias Audit pages read directly from session state and guard against premature access.

---

## Algorithms & Technical Detail

### 1. Text Preprocessing

**File:** `src/preprocessor.py`

Each resume and the JD undergo the same five-step pipeline before vectorisation:

1. **Lowercase** — `text.lower()`
2. **Punctuation & digit removal** — regex `[^a-z\s]` strips everything except letters and whitespace
3. **Tokenisation** — whitespace split
4. **Lemmatisation** — spaCy `en_core_web_sm` model; each token is replaced with its dictionary base form (e.g., *running* → *run*, *universities* → *university*)
5. **Stopword removal** — NLTK English stopword list (179 words); removes common words (*the*, *and*, *of*…) that carry no discriminating signal

The output is a single cleaned string fed to the TF-IDF vectoriser.

**Why lemmatise instead of stem?** Lemmatisation preserves linguistic validity (Porter stemmer would map *university* → *univers*, which is meaningless as a vocabulary term). It also improves recall: *managing* and *management* both reduce to *manage* and therefore match.

---

### 2. TF-IDF Vectorisation & Cosine Similarity

**File:** `src/scorer.py`

**TF-IDF (Term Frequency – Inverse Document Frequency)** converts text into numeric vectors where each dimension represents a vocabulary term, weighted by how important it is to that document relative to the corpus.

**Term Frequency (sublinear):**

```
TF(t, d) = 1 + log(count(t, d))    if count > 0
           0                         otherwise
```

`sublinear_tf=True` compresses high-frequency terms — a word appearing 100 times is not 100× more important than one appearing once.

**Inverse Document Frequency:**

```
IDF(t) = log((1 + N) / (1 + df(t))) + 1
```

where N = number of documents (JD + all resumes) and df(t) = documents containing term t. Rare terms get higher IDF; ubiquitous terms approach 1.

**Cosine Similarity:**

```
similarity(JD, resume_i) = (JD · resume_i) / (||JD|| × ||resume_i||)
```

The dot product of two L2-normalised TF-IDF vectors; 1.0 = identical, 0 = no shared terms.

**Raw score** = cosine similarity × 100. A strong match typically scores 25–45%; this is the inherent ceiling of TF-IDF cosine between two structurally different documents (a JD and a resume have different vocabulary distributions by nature).

---

### 3. Score Normalisation

**File:** `src/scorer.py`

Raw scores are batch-relative: the maximum raw score in the batch is used to normalise all others to 0–100%.

```
normalised_score(i) = (raw_score(i) / max(raw_scores)) × 100
```

**Why normalise?** A TF-IDF cosine ceiling of 35% would make fixed thresholds (e.g., "shortlist above 60%") meaningless. Normalisation makes the question relative: *"how does this candidate compare to the best available candidate in this batch?"* — which is the correct framing for screening.

The top scorer always gets 100%. All tiers and charts are based on the normalised score.

---

### 4. Rubric Scoring

**File:** `src/rubric_scorer.py`, `src/jd_builder.py`

When the user builds a JD via the Structured Form, a weighted rubric is generated alongside the JD text. The rubric evaluates each resume on explicit criteria and produces a 0–100 rubric score that is blended with the TF-IDF score.

**Knockout rules:** When RubricResult.knockout is true, classifier.py forces final_score = 0.0 and the candidate is classified as Not Suitable.

**Final score blending:**

```
final_score = 0.70 × normalised_TF-IDF + 0.30 × rubric_score
```

The 70/30 blend retains the content-based signal as dominant while allowing structured criteria to adjust rankings. If no structured JD is used (plain paste/upload), rubric scoring is skipped and the normalised TF-IDF score is used directly.

---

### 5. Tier Classification

**File:** `src/classifier.py`

Applied to the final score (or normalised score if no rubric):

| Tier | Threshold | Interpretation |
|---|---|---|
| Strong Match | ≥ 70% | Recommend for interview |
| Partial Match | 40–69% | Gap analysis shown; conditional recommendation |
| Not Suitable | < 40% | Below acceptable match threshold |

The thresholds are relative (applied to the normalised score), so they remain meaningful across batches of different sizes and quality.

---

### 6. Skill & Name Extraction

**File:** `src/extractor.py`

**Skill extraction:** A keyword list (`data/skills.txt`, ~200 entries) is matched against raw resume text using case-insensitive whole-word regex. Skills are grouped as matched (found in resume) vs missing (in JD skill list but absent from resume) for gap analysis.

**Name extraction:** spaCy's `en_core_web_sm` NER model labels named entities. The first `PERSON`-labelled entity in the resume is taken as the candidate name. This covers most Western and South Asian name formats that appear near the top of a resume. If no PERSON entity is found, the filename is used as a fallback.

---

### 7. Bias Audit

**File:** `src/bias_audit.py`

See [Bias Audit Deep-Dive](#bias-audit-deep-dive) for full detail.

---

## Bias Audit Deep-Dive

The audit runs automatically after every analysis and covers four dimensions.

### Gender Dimension

**Inference (two-stage):**

1. **Name lookup** — a curated dictionary of ~60 common first names mapped to Male/Female. The candidate's extracted first name is looked up; if found, inference is marked as `name_lookup`.
2. **Pronoun scan fallback** — if the name is not in the dictionary, the raw resume text is scanned for gendered pronouns: `he/him/his` → Male, `she/her/hers` → Female. A minimum of 2 occurrences and a clear majority (no tie) is required. Method marked as `pronoun_scan`.
3. If neither method yields a result, the candidate is marked `Unknown`.

**Condition for auditing:** The gender audit only activates when the JD's gender preference is set to "No preference". If the JD explicitly targets a gender (for a diversity hire programme), auditing is skipped.

**Disparity detection:**

- Mean normalised score is computed per inferred gender group.
- Groups with fewer than 5 candidates are flagged as low-sample (estimates unreliable).
- **Mann-Whitney U test** is used when exactly two known groups exist; **Kruskal-Wallis H test** is used for three or more. Both are non-parametric (no normality assumption).
- **pp-gap** = absolute difference between the two highest group means, in percentage points.
- **Disparate Impact Ratio (DIR)** = min(group_mean) / max(group_mean). A DIR < 0.80 is the configured threshold.
- Bias is first considered when pp-gap > 10 OR DIR < 0.80; if a p-value is available and p ≥ 0.05, the bias flag is cleared. In practice, the effective statistical threshold is p < 0.05.

**Counterfactual correction (when bias detected):**

1. All candidate name tokens are replaced with `[CANDIDATE]` in the raw text.
2. The masked resumes are re-scored against the JD using the same TF-IDF pipeline.
3. The corrected score = average of original score and masked score.
4. Up to 3 iterative correction rounds are applied until the per-candidate delta < 2 pp.

The correction neutralises any name-vocabulary leakage into the cosine similarity signal.

### Career Gap Dimension

**Condition for auditing:** Active only when the JD max career gap is set to 0 (no restriction).

**Gap detection:** Regex patterns search for employment dates and compute gap months between roles. Candidates with gaps > 6 months are classified as "gap" group; others as "no-gap".

**Statistical test:** Mann-Whitney U on two groups (gap / no-gap). Flags if the no-gap group scores systematically higher, indicating an unintended algorithmic penalty for career breaks.

### Institution Tier Dimension

A curated list of top-tier institutions (QS World Rankings top-200 plus prominent Indian institutions — IITs, IIMs, NITs) is matched against raw resume text. Candidates are grouped as Tier 1 / Other.

Flags if Tier 1 graduates score significantly higher after controlling for content similarity, suggesting the JD vocabulary may inadvertently favour elite institution terminology.

### Socio-Cultural Dimension (India-specific)

A surname lookup database maps last names to:
- **Region proxy**: North / South / East / West India
- **Religion proxy**: Hindu / Muslim / Christian / Sikh / Other
- **Caste group proxy**: General / OBC / SC / ST / Unknown

Each sub-dimension runs an independent Kruskal-Wallis test. If caste-group disparity is detected, the audit surfaces India's constitutional reservation framework (Articles 15 and 16) as context.

**Important caveat:** Surname proxies are statistical approximations — they will be incorrect for individual candidates. The purpose is systemic pattern detection across a batch, not individual assessment.

---

## UI & Navigation

The app uses Streamlit's multi-page navigation (`st.navigation()` / `st.Page()`) with three routes:

| Route | URL | Content |
|---|---|---|
| Upload & Analyse | `/upload` | JD builder, resume upload, analysis pipeline with live progress |
| Results | `/results` | Stat cards, ranked table, score analytics, gap analysis, CSV export |
| Bias Audit | `/bias` | 4-step audit (what was checked, what was found, what was corrected, score distribution) |

A persistent top navigation bar (rendered in every page via `app.py`) uses `st.page_link()` for client-side navigation that preserves session state. Results and Bias Audit are disabled until analysis has been run.

**Design system:** flat, sharp-cornered enterprise design (no border-radius), navy `#0c1a2e` brand + gold `#C8970A` accent, page background `#f2f5f9`.

**Authentication:** login-gated via `streamlit-authenticator`; credentials stored as bcrypt hashes in `config.yaml`; 30-day session cookies.

---

## Installation

**Python 3.11 is required.** spaCy 3.7.4 (blis dependency) cannot build on Python 3.12+.

```bash
py -3.11 -m pip install -r requirements.txt
```

Install spaCy language model (one-time):

```bash
py -3.11 -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

Create the first user account:

```bash
py -3.11 scripts/create_user.py --add hr_admin --password yourpassword --name "HR Admin" --email hr@company.com
```

---

## Usage

```bash
py -3.11 -m streamlit run app.py
```

The app hot-reloads on file save. Open http://localhost:8501.

### Built-in demo (quickest)

Click **Load Demo Data** on the Upload page. Loads 15 pre-built resume PDFs and a Software Engineer JD from `data/demo_resumes/` and runs the full pipeline immediately.

### Upload your own files

1. Build a JD using the Structured Form (select profession, seniority, skills) **or** paste/upload a plain-text or PDF JD.
2. Upload resume files or click **Browse folder…** to pick a directory.
3. Click **Run Analysis** — a live progress indicator shows extraction → scoring → audit stages.
4. Navigate to **Results** or **Bias Audit** via the top navbar.

### Download additional resumes

```bash
# 50 diverse resumes from HuggingFace
py -3.11 scripts/download_samples.py

# Custom role mix
py -3.11 scripts/download_samples.py \
    --role-mix "Data Scientist:20,ML Engineer:15,Product Manager:15" \
    --dest ~/Downloads/resumes --format pdf
```

---

## Project Structure

```
auto-resume-sifter/
│
├── app.py                      Entry point — auth, CSS, navbar, st.navigation()
│
├── pages/
│   ├── upload.py               JD builder + resume upload + analysis pipeline
│   ├── results.py              Ranked results table + analytics + gap analysis
│   └── bias.py                 4-step bias audit UI
│
├── src/
│   ├── reader.py               PDF (pdfplumber) + TXT text extraction
│   ├── preprocessor.py         5-step NLP cleaning pipeline
│   ├── extractor.py            spaCy NER name extraction + skill keyword matching
│   ├── scorer.py               TF-IDF vectoriser (scikit-learn) + cosine similarity
│   ├── classifier.py           Tier classification + Candidate dataclass
│   ├── rubric_scorer.py        Weighted structured criteria scorer
│   ├── bias_audit.py           Multi-dimensional bias detection & correction
│   ├── jd_builder.py           JD text generator + rubric configuration
│   └── ui_helpers.py           Shared CSS block, nav helpers, formatting functions
│
├── data/
│   ├── demo_resumes/           15 formatted PDF resumes + jd.pdf (HuggingFace source)
│   ├── skills.txt              ~200-entry skill keyword list
│   └── demo_jd.txt             Fallback job description (plain text)
│
├── scripts/
│   ├── download_samples.py     HuggingFace resume downloader + PDF renderer
│   └── create_user.py          CLI tool to add/remove app users
│
├── tests/
│   ├── test_preprocessor.py    Unit tests for cleaning pipeline
│   ├── test_scorer.py          Unit tests for TF-IDF scoring
│   └── test_classifier.py      Unit tests for tier classification
│
├── config.yaml                 Encrypted user credentials (bcrypt hashes)
├── requirements.txt
└── README.md
```

---

## Testing

```bash
py -3.11 -m pytest
```

23 unit tests covering:
- **Preprocessor**: lowercasing, punctuation removal, lemmatisation, stopword removal, empty-input handling
- **Scorer**: cosine similarity correctness, normalisation, edge cases (single resume, identical documents)
- **Classifier**: tier boundary conditions (exactly 70%, exactly 40%, fractional scores)

---

## Limitations

- **TF-IDF cosine ceiling**: Raw scores rarely exceed 40–45% even for a strong match, because a JD and resume have structurally different vocabulary distributions. Normalisation compensates for this but does not change the underlying signal.
- **Vocabulary sensitivity**: Candidates who use the exact keywords from the JD score higher than equally-qualified candidates who paraphrase. This is a known limitation of bag-of-words models.
- **Bias audit sample size**: Statistical reliability requires n > 30 per group. At demo scale (15 resumes), results are indicative only. A production deployment should use Fairlearn or AI Fairness 360 with continuous monitoring.
- **Socio-cultural proxies**: Surname-to-group mappings are approximate and will be wrong for diaspora names, transliterated variations, and cross-community marriages.
- **Gender inference**: The name dictionary covers ~60 entries. The pronoun-scan fallback requires explicit pronoun use in the resume text, which many candidates omit.
- **English only**: The preprocessing pipeline (spaCy `en_core_web_sm`, NLTK English stopwords) is designed for English-language resumes and JDs only.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | 1.58 | Multi-page web UI framework |
| streamlit-authenticator | 0.3.2 | Login gate + bcrypt credential management |
| spacy + en_core_web_sm | 3.7.4 / 3.7.1 | NER for name extraction; lemmatisation |
| scikit-learn | 1.4 | TF-IDF vectoriser + cosine similarity |
| pdfplumber | 0.11 | PDF text extraction |
| nltk | 3.8 | English stopword list |
| plotly | 5.22 | Interactive score and audit charts |
| pandas | 2.2 | DataFrame operations + CSV export |
| datasets | 2.19 | HuggingFace dataset downloads |
| reportlab | 4.2 | PDF generation for demo data |
| scipy | 1.13 | Mann-Whitney U + Kruskal-Wallis statistical tests |
| pyyaml | 6.0 | YAML config parsing |

---

## Data Source

Demo resumes are sourced from [`AzharAli05/Resume-Screening-Dataset`](https://huggingface.co/datasets/AzharAli05/Resume-Screening-Dataset) on HuggingFace — 10,174 rows with Role, Resume, and Job_Description columns. The download script filters by role and renders resumes as formatted PDFs using ReportLab.
