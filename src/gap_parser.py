"""
Enriches Candidate objects with:
  - career_gap_months     : total inferred employment gap in months
  - institution_tier      : "Top" | "Other" | "Unknown"
  - region_proxy          : North | South | East | West | Northeast | Unknown
  - religion_proxy        : Hindu | Muslim | Sikh | Christian | Jain | Zoroastrian | Other | Unknown
  - caste_group_proxy     : General | OBC | SC | ST | Unknown

All values are approximations from resume text parsing.
"""

from __future__ import annotations

import csv
import pathlib
import re
from calendar import month_abbr
from functools import lru_cache

_DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
_INSTITUTION_FILE = _DATA_DIR / "institution_tiers.txt"
_SURNAME_FILE = _DATA_DIR / "indian_surname_proxy.csv"

# ── Institution tier lookup ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_institution_tiers() -> list[str]:
    if not _INSTITUTION_FILE.exists():
        return []
    lines = []
    for line in _INSTITUTION_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line.lower())
    return lines


def detect_institution_tier(text: str) -> str:
    """Return 'Top', 'Other', or 'Unknown'."""
    tiers = _load_institution_tiers()
    if not tiers:
        return "Unknown"
    t = text.lower()
    for institution in tiers:
        if institution in t:
            return "Top"
    # If any education section present, assume "Other"; else Unknown
    if re.search(r"\b(university|college|institute|b\.?tech|b\.?e\.|bsc|bca|mtech|msc|mca|phd)\b", t, re.IGNORECASE):
        return "Other"
    return "Unknown"


# ── Surname proxy lookup ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_surname_map() -> dict[str, dict[str, str]]:
    """Return {surname: {region, religion, caste_group}}."""
    result: dict[str, dict[str, str]] = {}
    if not _SURNAME_FILE.exists():
        return result
    with open(_SURNAME_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        for row in reader:
            sn = row.get("surname", "").strip().lower()
            if sn:
                result[sn] = {
                    "region": row.get("region", "Unknown").strip(),
                    "religion": row.get("religion", "Unknown").strip(),
                    "caste_group": row.get("caste_group", "Unknown").strip(),
                }
    return result


def detect_surname_proxy(full_name: str | None) -> dict[str, str]:
    """Extract last name and look up region/religion/caste_group."""
    _default = {"region": "Unknown", "religion": "Unknown", "caste_group": "Unknown"}
    if not full_name:
        return _default
    parts = full_name.strip().split()
    if len(parts) < 2:
        return _default
    last = parts[-1].lower()
    surname_map = _load_surname_map()
    return surname_map.get(last, _default)


# ── Career gap detection ───────────────────────────────────────────────────────

_MONTH_NAMES = {m.lower(): i for i, m in enumerate(month_abbr) if m}
_MONTH_NAMES.update({
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
})

_DATE_RANGE_PATTERNS = [
    # "Jan 2020 – Mar 2021" or "Jan 2020 - Mar 2021"
    r"(?P<sm>[A-Za-z]{3,9})\s+(?P<sy>\d{4})\s*[–—\-]+\s*(?P<em>[A-Za-z]{3,9})\s+(?P<ey>\d{4})",
    # "2019 – 2022"
    r"(?P<sy2>\d{4})\s*[–—\-]+\s*(?P<ey2>\d{4})",
    # "Jan 2020 – Present" / "Jan 2020 – Current"
    r"(?P<sm2>[A-Za-z]{3,9})\s+(?P<sy3>\d{4})\s*[–—\-]+\s*(?:present|current|ongoing|now)",
    # "2020 – Present"
    r"(?P<sy4>\d{4})\s*[–—\-]+\s*(?:present|current|ongoing|now)",
]


def _month_to_int(month_str: str) -> int:
    return _MONTH_NAMES.get(month_str.lower()[:3], 0) or _MONTH_NAMES.get(month_str.lower(), 0)


def _to_months(year: int, month: int) -> int:
    return year * 12 + month


def detect_career_gap(text: str, current_year: int = 2025, current_month: int = 6) -> int:
    """
    Detect total employment gap months from a resume text.

    Returns 0 if no date ranges are found or no gap exists.
    The gap is the total time between consecutive employment periods.
    """
    periods: list[tuple[int, int]] = []  # (start_months, end_months)

    for pattern in _DATE_RANGE_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            gd = m.groupdict()
            try:
                if gd.get("sm") and gd.get("sy") and gd.get("em") and gd.get("ey"):
                    start = _to_months(int(gd["sy"]), _month_to_int(gd["sm"]) or 1)
                    end = _to_months(int(gd["ey"]), _month_to_int(gd["em"]) or 12)
                elif gd.get("sy2") and gd.get("ey2"):
                    start = _to_months(int(gd["sy2"]), 1)
                    end = _to_months(int(gd["ey2"]), 12)
                elif gd.get("sm2") and gd.get("sy3"):
                    start = _to_months(int(gd["sy3"]), _month_to_int(gd["sm2"]) or 1)
                    end = _to_months(current_year, current_month)
                elif gd.get("sy4"):
                    start = _to_months(int(gd["sy4"]), 1)
                    end = _to_months(current_year, current_month)
                else:
                    continue
                if start > 0 and end >= start:
                    periods.append((start, end))
            except (ValueError, KeyError):
                continue

    if not periods:
        return 0

    # Sort by start date and merge overlapping periods
    periods.sort(key=lambda x: x[0])
    merged: list[tuple[int, int]] = []
    for start, end in periods:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    if len(merged) < 2:
        return 0

    # Sum gaps between consecutive periods
    total_gap = 0
    for i in range(1, len(merged)):
        gap = merged[i][0] - merged[i - 1][1]
        if gap > 0:
            total_gap += gap

    return total_gap


# ── Main enrichment entry point ───────────────────────────────────────────────

def enrich_candidate(candidate) -> None:
    """
    Enrich a Candidate object in-place with gap, institution, and surname proxy data.
    Modifies: career_gap_months, institution_tier, region_proxy, religion_proxy, caste_group_proxy.
    """
    text = candidate.raw_text or ""
    candidate.career_gap_months = detect_career_gap(text)
    candidate.institution_tier = detect_institution_tier(text)
    proxy = detect_surname_proxy(candidate.name)
    candidate.region_proxy = proxy["region"]
    candidate.religion_proxy = proxy["religion"]
    candidate.caste_group_proxy = proxy["caste_group"]
