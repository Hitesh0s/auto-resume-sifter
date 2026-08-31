# Auto Resume Sifter

Streamlit web app that screens and ranks candidate resumes against a job description using TF-IDF cosine similarity, optional rubric scoring, and a multi-dimensional bias audit. Built for a university final-year project.

## Running the app

```
py -3.11 -m streamlit run app.py
```

**Python version: must use 3.11.** The default system Python is 3.14, which cannot build `spacy==3.7.4` (blis compilation fails). Always invoke with `py -3.11`.

The app hot-reloads on file save; no restart needed for code changes.

## Key dependencies

| Package | Purpose |
|---|---|
| streamlit 1.58 | Multi-page UI framework |
| streamlit-authenticator 0.3.2 | Login gate + bcrypt credential management |
| spacy 3.7.4 + en_core_web_sm | NER for name extraction + lemmatisation |
| scikit-learn 1.4 | TF-IDF vectoriser + cosine similarity |
| pdfplumber 0.11 | PDF text extraction |
| nltk 3.8 | Stopword list for preprocessing |
| plotly 5.22 | Score distribution + audit charts |
| scipy 1.13 | Mann-Whitney U + Kruskal-Wallis statistical tests |
| datasets 2.19 | HuggingFace dataset downloads |
| reportlab 4.2 | PDF rendering for demo data generation |

Install: `py -3.11 -m pip install -r requirements.txt`

spaCy model (one-time): `py -3.11 -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl`

## Project structure

```
app.py                      Entry point — auth, CSS, top navbar, st.navigation()
pages/
  upload.py                 JD builder + resume upload + analysis pipeline
  results.py                Ranked results table + analytics + gap analysis
  bias.py                   4-step bias audit UI
src/
  reader.py                 PDF + TXT text extraction (pdfplumber)
  preprocessor.py           5-step NLP cleaning pipeline
  extractor.py              Name extraction (spaCy NER) + skill keyword matching
  scorer.py                 TF-IDF cosine similarity scoring + normalisation
  classifier.py             Tier classification + Candidate dataclass
  rubric_scorer.py          Weighted structured criteria scorer
  bias_audit.py             Multi-dimensional bias detection + correction
  jd_builder.py             Structured JD text generator + rubric config
  ui_helpers.py             Shared CSS block, navbar helpers, formatting functions
data/
  demo_resumes/             15 HuggingFace resumes as formatted PDFs + jd.pdf
  skills.txt                ~200-entry skill keyword list
  demo_jd.txt               Legacy fallback JD
scripts/
  download_samples.py       HuggingFace resume downloader + PDF renderer
  create_user.py            CLI tool to add/remove user accounts
tests/                      pytest unit tests (preprocessor, scorer, classifier)
config.yaml                 Encrypted user credentials (bcrypt hashes)
```

## Authentication

User accounts are managed via `scripts/create_user.py`:
```
py -3.11 scripts/create_user.py --add username --password pass --name "Full Name" --email x@y.com
```

Credentials are bcrypt-hashed and stored in `config.yaml`. `streamlit-authenticator` manages 30-day session cookies. The login page shows a dark-gradient branding panel above the form.

## Scoring approach

### Step 1 — Text Preprocessing (`src/preprocessor.py`)

Each resume and the JD go through the same five-step pipeline:
1. Lowercase
2. Punctuation and digit removal (regex `[^a-z\s]`)
3. Tokenisation (whitespace split)
4. Lemmatisation — spaCy `en_core_web_sm`; maps inflected forms to dictionary root
5. Stopword removal — NLTK English stopword list (179 words)

Lemmatisation is preferred over stemming: it preserves valid vocabulary terms (e.g., *manage* from *management* and *managing*) whereas Porter stemmer produces invalid roots (*univers* from *university*).

### Step 2 — TF-IDF Vectorisation (`src/scorer.py`)

scikit-learn `TfidfVectorizer` with:
- `ngram_range=(1,1)` — unigrams only; bigrams inflate vocabulary ~4× and collapse cosine scores below 15% for small corpora
- `sublinear_tf=True` — `TF = 1 + log(count)` to dampen high-frequency terms
- IDF computed over the full corpus (JD + all uploaded resumes), not per-pair

Vectors are L2-normalised, so cosine similarity reduces to a dot product.

### Step 3 — Cosine Similarity → Raw Score

```
raw_score(i) = dot(JD_vec, resume_vec_i) × 100
```

Typical ceiling: 25–45% for a strong match. Two structurally different documents (JD vs resume) share only a fraction of vocabulary even when well-matched.

### Step 4 — Score Normalisation

```
normalised_score(i) = (raw_score(i) / max(raw_scores)) × 100
```

The top scorer in the batch always gets 100%. Tiers and all charts use the normalised score.

### Step 5 — Rubric Scoring (`src/rubric_scorer.py`)

Activated when a JD is built using the Structured Form. Criteria and default weights:

| Criterion | Weight |
|---|---|
| Required skills coverage | 40% |
| Education floor | 20% |
| Experience range | 15% |
| Career gap | 10% |
| Soft skills | 10% |
| Relocation | 5% |

Knockout: if the minimum degree is not met, rubric score is capped at 20%.

Final score blending (when rubric available):
```
final_score = 0.70 × normalised_TF-IDF + 0.30 × rubric_score
```

### Step 6 — Tier Classification (`src/classifier.py`)

Applied to final_score (or normalised score if no rubric):
- **Strong Match** ≥ 70%
- **Partial Match** ≥ 40%
- **Not Suitable** < 40%

Both raw and normalised scores are stored on the `Candidate` dataclass and exported in CSV.

## Bias audit (`src/bias_audit.py`)

Four independent audit dimensions, each with its own activation condition:

### Gender
- **Inference**: first-name dictionary lookup (~60 entries); pronoun scan fallback (`he/him/his` → Male, `she/her/hers` → Female; requires ≥ 2 occurrences, clear majority)
- **Condition**: JD gender preference = "No preference"
- **Statistical test**: Mann-Whitney U (2 groups) or Kruskal-Wallis H (3+ groups); p < 0.10 threshold
- **Disparity metric**: pp-gap (percentage-point difference between group means) > 10; Disparate Impact Ratio (DIR = min_mean / max_mean) < 0.80
- **Counterfactual correction**: name tokens replaced with `[CANDIDATE]`, re-scored, corrected = average(original, masked); up to 3 iterative rounds until delta < 2 pp

### Career Gap
- **Condition**: JD max gap = 0 (no restriction set)
- Regex detects employment gaps > 6 months; Mann-Whitney U on gap / no-gap groups

### Institution Tier
- Curated top-tier list (QS top-200 + IITs, IIMs, NITs); candidates grouped Tier 1 / Other
- Kruskal-Wallis test on score distributions

### Socio-cultural (India)
- Surname lookup database → region proxy, religion proxy, caste group proxy
- Three independent Kruskal-Wallis tests (one per sub-dimension)
- Caste disparity results reference India's constitutional reservation framework (Articles 15, 16)

`AuditResult` fields: `bias_detected`, `group_stats`, `corrected`, `candidate_deltas`, `low_sample_groups`, `inference_rows`, `dimension_results`.

## Multi-page navigation

`app.py` uses `st.navigation([st.Page(...), ...], position="hidden")` to register three routes:
- `/upload` — Upload & Analyse (default)
- `/results` — Results
- `/bias` — Bias Audit

A persistent top bar is rendered in `app.py` before `pg.run()` using `st.columns()` + `st.page_link()`. Page links use Streamlit's React Router client-side navigation, preserving `st.session_state` across page switches. Results and Bias Audit show `disabled=True` until `st.session_state.ready` is set by the analysis pipeline.

## UI layout

**Upload & Analyse page (`pages/upload.py`):**
- Demo banner (Load Demo Data button)
- JD input — two tabs: Structured Form | Paste/Upload
- Resume input — two tabs: Upload files | Load from folder
- Run Analysis → `st.status()` spinner with `st.progress()` bar inside (4 stages: extract → score → audit → done)
- Session state stored: `ranked`, `audit`, `jd_skills`, `rubric`, `vectorizer`, `tfidf_matrix`, `ready`

**Results page (`pages/results.py`):**
- Guard: `if not ready → stop`
- Stat cards (total, strong, partial, not suitable, top scorer name)
- Tier filter + results table (raw score, normalised score, tier badge, matched skills)
- Score analytics — 3 Plotly tabs: score distribution, top discriminating TF-IDF terms, skill heatmap
- Top scorer spotlight card (gold border)
- Gap analysis — expanders for each Partial Match candidate with skill tags + recommendation
- CSV export button

**Bias Audit page (`pages/bias.py`):**
- Guard: `if not ready → stop`
- Step 1: 4 dimension cards (audited / skipped / bias detected)
- Step 2: per-dimension stat tables in expanders (gender, career gap, institution tier, socio-cultural)
- Step 3: name-masking correction delta table + before/after bar chart
- Step 4: score distribution by inferred gender
- Methodology + limitations expander

## Design system

| Token | Value | Use |
|---|---|---|
| Navy dark | `#0c1a2e` | Brand text, nav divider, headings |
| Navy mid | `#1E3A5F` | Buttons, stat card top border, links |
| Gold | `#C8970A` | Active nav indicator, section heading underline, top scorer |
| Surface | `#f2f5f9` | Page background |
| Card | `#ffffff` | Stat cards, expanders, table rows |
| Border | `#dde4ee` | All borders |
| Muted | `#5e738a` | Subtitle text, labels |

No border-radius anywhere (`*, *::before, *::after { border-radius: 0 !important; }`); badges use 2px only.

## Demo data

**Built-in (15 resumes):** Click **Load Demo Data** on the Upload page. Resumes and JD loaded from `data/demo_resumes/`. Auto-detects `jd.pdf` or `jd.txt`, falling back to `data/demo_jd.txt`.

**Regenerate built-in demo data:**
```
py -3.11 scripts/download_samples.py \
    --dest data/demo_resumes \
    --role-mix "Software Engineer:5,Data Scientist:5,Human Resources Specialist:5" \
    --format pdf
```

**HuggingFace (50+ resumes):**
```
py -3.11 scripts/download_samples.py
```
Saves to `~/Downloads/auto-resume-sifter-demo/resumes/`. Use **Load from folder** in the app.

Source dataset: `AzharAli05/Resume-Screening-Dataset` (10,174 rows, Role + Resume + Job_Description columns).

## Tests

```
py -3.11 -m pytest
```

23 unit tests across 3 files:
- `tests/test_preprocessor.py` — cleaning pipeline steps, empty input, lemmatisation
- `tests/test_scorer.py` — cosine correctness, normalisation, edge cases
- `tests/test_classifier.py` — tier boundary conditions (70%, 40%, fractional)
