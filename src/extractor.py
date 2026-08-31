import functools
import pathlib
import re

import spacy


@functools.lru_cache(maxsize=1)
def _nlp():
    return spacy.load("en_core_web_sm")


@functools.lru_cache(maxsize=1)
def _skills_list() -> list[str]:
    skills_path = pathlib.Path(__file__).parent.parent / "data" / "skills.txt"
    skills = []
    for line in skills_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            skills.append(line)
    return skills

_NAME_REJECT = re.compile(r"[@\[\]{}()<>0-9]")

def _looks_like_name(candidate: str) -> bool:
    """Reject entity/line matches that aren't shaped like a human name
    (emails, URLs, markdown-link remnants, job titles with digits, etc.)."""
    if not candidate or not (2 <= len(candidate) <= 60):
        return False
    if _NAME_REJECT.search(candidate):
        return False
    if "mailto" in candidate.lower() or "http" in candidate.lower():
        return False
    return True


def extract_name(text: str) -> str | None:
    """Return the first PERSON entity found in the first 200 characters."""
    snippet = text[:200]
    doc = _nlp()(snippet)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            candidate = ent.text.strip()
            if _looks_like_name(candidate):
                return candidate
    # Fallback: first non-empty line that looks like a name (2-4 capitalized words)
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^([A-Z][a-z]+ ){1,3}[A-Z][a-z]+$", line) and _looks_like_name(line):
            return line
    return None


def extract_skills(text: str) -> list[str]:
    """Return deduplicated skills found in text via case-insensitive match."""
    text_lower = text.lower()
    found = []
    seen: set[str] = set()
    for skill in _skills_list():
        key = skill.lower()
        if key in seen:
            continue
        # Word-boundary match to avoid partial hits (e.g. "R" in "React")
        pattern = r"(?<![a-zA-Z0-9/.])" + re.escape(key) + r"(?![a-zA-Z0-9/.])"
        if re.search(pattern, text_lower):
            found.append(skill)
            seen.add(key)
    return found
