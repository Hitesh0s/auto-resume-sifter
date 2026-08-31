"""
Prepare demo resume data for Auto Resume Sifter.

Downloads 15 sample resumes from a public resume dataset and saves them as
TXT files in data/demo_resumes/ so the app has realistic input for the demo.

Tier distribution (tested against data/demo_jd.txt — Software Engineer):
  5 x Python Developer  →  Strong Match  (high overlap with JD)
  5 x Data Science      →  Partial Match (Python overlap, but less Django/Docker)
  5 x HR                →  Not Suitable  (no technical overlap)

Usage
-----
Default (HuggingFace, no credentials needed):
    python scripts/prepare_demo_data.py

With a locally downloaded Kaggle CSV:
    python scripts/prepare_demo_data.py --csv data/UpdatedResumeDataSet.csv

The Kaggle dataset (Category + Resume columns) can be downloaded from:
    https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import textwrap

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "data" / "demo_resumes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace dataset IDs to try in order (same underlying data, different mirrors)
_HF_CANDIDATES = [
    ("likecodin/resume_dataset", "train"),
    ("InfusedAI/Resume_DS_English", "train"),
]

# Which categories to pull and how many per tier
TIER_PLAN: list[tuple[str, str, int]] = [
    ("strong",     "Python Developer", 5),
    ("partial",    "Data Science",     5),
    ("unsuitable", "HR",               5),
]


# ── loaders ──────────────────────────────────────────────────────────────────

def _load_hf() -> dict[str, list[str]] | None:
    """Try each HuggingFace dataset candidate; return category→resumes map."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' is not installed. Run: pip install datasets")
        return None

    for dataset_id, split in _HF_CANDIDATES:
        try:
            print(f"Trying HuggingFace: {dataset_id} / {split} ...")
            ds = load_dataset(dataset_id, split=split, trust_remote_code=False)
            cols = ds.column_names

            # Normalise column names — different mirrors use different casing
            cat_col = next((c for c in cols if c.lower() == "category"), None)
            res_col = next((c for c in cols if c.lower() == "resume"), None)
            if not cat_col or not res_col:
                print(f"  Skipping — expected 'Category' and 'Resume' columns, got: {cols}")
                continue

            by_cat: dict[str, list[str]] = {}
            for row in ds:
                by_cat.setdefault(row[cat_col], []).append(row[res_col])
            print(f"  Loaded {sum(len(v) for v in by_cat.values())} resumes "
                  f"across {len(by_cat)} categories.\n")
            return by_cat

        except Exception as exc:
            print(f"  Failed: {exc}")

    return None


def _load_csv(csv_path: str) -> dict[str, list[str]] | None:
    """Load from a locally downloaded Kaggle CSV."""
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: 'pandas' is not installed. Run: pip install pandas")
        return None

    path = pathlib.Path(csv_path)
    if not path.exists():
        print(f"ERROR: CSV file not found at '{csv_path}'")
        return None

    df = pd.read_csv(path)

    # Tolerate different column name cases
    df.columns = [c.strip() for c in df.columns]
    cat_col = next((c for c in df.columns if c.lower() == "category"), None)
    res_col = next((c for c in df.columns if c.lower() == "resume"), None)
    if not cat_col or not res_col:
        print(f"ERROR: Expected 'Category' and 'Resume' columns. Found: {list(df.columns)}")
        return None

    by_cat: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        by_cat.setdefault(str(row[cat_col]), []).append(str(row[res_col]))
    print(f"Loaded {len(df)} resumes across {len(by_cat)} categories from CSV.\n")
    return by_cat


# ── writer ────────────────────────────────────────────────────────────────────

def _save(text: str, tier: str, index: int, category: str) -> None:
    slug = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")
    filename = OUTPUT_DIR / f"resume_{tier}_{index:02d}_{slug}.txt"
    # Wrap long lines so pdfplumber-extracted equivalents look natural
    wrapped = "\n".join(
        textwrap.fill(line, width=100) if len(line) > 100 else line
        for line in text.splitlines()
    )
    filename.write_text(wrapped, encoding="utf-8")
    print(f"  Saved: {filename.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(csv_path: str | None = None) -> None:
    if csv_path:
        by_cat = _load_csv(csv_path)
    else:
        by_cat = _load_hf()

    if by_cat is None:
        print("\nCould not load dataset. Options:")
        print("  1. Install datasets:  pip install datasets")
        print("     Then retry:        python scripts/prepare_demo_data.py")
        print()
        print("  2. Download the Kaggle CSV manually:")
        print("     https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset")
        print("     Place it at:  data/UpdatedResumeDataSet.csv")
        print("     Then run:     python scripts/prepare_demo_data.py --csv data/UpdatedResumeDataSet.csv")
        sys.exit(1)

    print(f"Available categories: {sorted(by_cat.keys())}\n")

    saved = 0
    for tier, category, count in TIER_PLAN:
        pool = by_cat.get(category, [])
        if not pool:
            # Try case-insensitive match
            pool = next(
                (v for k, v in by_cat.items() if k.lower() == category.lower()),
                [],
            )
        if not pool:
            print(f"WARNING: No resumes found for category '{category}'. Skipping.\n")
            continue

        print(f"Saving {count} '{category}' resumes (tier: {tier}):")
        for i, text in enumerate(pool[:count], 1):
            _save(text, tier, i, category)
            saved += 1
        print()

    print(f"Done — {saved} demo resumes written to: {OUTPUT_DIR}\n")
    print("To launch the app:  streamlit run app.py")
    print("Then upload resumes from:  data/demo_resumes/")
    print("And use JD from:           data/demo_jd.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Auto Resume Sifter demo data.")
    parser.add_argument(
        "--csv",
        metavar="PATH",
        help="Path to a locally downloaded Kaggle CSV (Category + Resume columns).",
        default=None,
    )
    args = parser.parse_args()
    main(csv_path=args.csv)
