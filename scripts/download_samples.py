"""
Download sample resumes and a matching job description from HuggingFace and save
them as formatted PDFs (or plain text).

Source: AzharAli05/Resume-Screening-Dataset (Role, Resume, Job_Description columns)

Usage:
    py -3.11 scripts/download_samples.py
    py -3.11 scripts/download_samples.py --role "Software Engineer" --count 50
    py -3.11 scripts/download_samples.py --list-roles
    py -3.11 scripts/download_samples.py --dest data/demo_resumes --count 15 \\
        --role-mix "Software Engineer:5,Data Scientist:5,HR Manager:5"
    py -3.11 scripts/download_samples.py --format txt   # save as plain text
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import textwrap
from collections import Counter

DATASET_ID = "AzharAli05/Resume-Screening-Dataset"
_DEFAULT_OUT_DIR = pathlib.Path.home() / "Downloads" / "auto-resume-sifter-demo"
DEFAULT_COUNT = 50

_PREFERRED_ROLES = [
    "Software Engineer",
    "Data Scientist",
    "Data Analyst",
    "Machine Learning Engineer",
    "DevOps Engineer",
    "Backend Developer",
    "Full Stack Developer",
    "Product Manager",
    "Business Analyst",
    "HR Manager",
]

# Navy colour used in PDF headings
_NAVY = "#1E3A5F"
_DARK = "#1A1A2E"
_GREY = "#64748B"


# ── PDF renderer ─────────────────────────────────────────────────────────────

def _render_pdf(text: str, path: pathlib.Path) -> None:
    """Render resume/JD plain text as a formatted PDF using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        from reportlab.lib.units import cm
    except ImportError:
        print("ERROR: reportlab not installed. Run: py -3.11 -m pip install reportlab")
        sys.exit(1)

    lines = [l.rstrip() for l in text.splitlines()]
    # Trim leading blank lines
    while lines and not lines[0].strip():
        lines.pop(0)

    NAVY_C = HexColor(_NAVY)
    DARK_C = HexColor(_DARK)
    GREY_C = HexColor(_GREY)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    name_style = ParagraphStyle(
        "Name", fontSize=18, fontName="Helvetica-Bold",
        textColor=NAVY_C, spaceAfter=3,
    )
    heading_style = ParagraphStyle(
        "Heading", fontSize=11, fontName="Helvetica-Bold",
        textColor=NAVY_C, spaceBefore=10, spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "Body", fontSize=10, fontName="Helvetica",
        textColor=DARK_C, spaceAfter=2, leading=14,
    )

    story = []
    first_used = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 5))
            continue

        # Escape reportlab markup characters
        safe = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        if not first_used:
            story.append(Paragraph(safe, name_style))
            story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY_C, spaceAfter=6))
            first_used = True
        elif stripped.isupper() and 3 <= len(stripped) <= 50:
            # ALL-CAPS line → section header
            story.append(Paragraph(safe, heading_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=GREY_C, spaceAfter=4))
        else:
            story.append(Paragraph(safe, body_style))

    doc.build(story)


# ── Text helpers ──────────────────────────────────────────────────────────────

def _wrap(text: str) -> str:
    lines = []
    for line in str(text).splitlines():
        lines.append(textwrap.fill(line, width=100) if len(line) > 100 else line)
    return "\n".join(lines)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _load_dataset():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Loading {DATASET_ID} from HuggingFace...")
    ds = load_dataset(DATASET_ID, split="train", trust_remote_code=False)
    print(f"  {len(ds)} rows, columns: {ds.column_names}\n")
    return ds


def _build_index(ds) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for i, row in enumerate(ds):
        role = str(row.get("Role", "Unknown")).strip()
        index.setdefault(role, []).append(i)
    return index


def _save_resume(text: str, role: str, seq: int, out_dir: pathlib.Path, fmt: str) -> None:
    ext = "pdf" if fmt == "pdf" else "txt"
    fname = out_dir / f"resume_{seq:02d}_{_slug(role)}.{ext}"
    if fmt == "pdf":
        _render_pdf(text, fname)
    else:
        fname.write_text(_wrap(text), encoding="utf-8")
    print(f"  {fname.name}")


# ── Row selection ─────────────────────────────────────────────────────────────

def _pick_rows_uniform(
    index: dict[str, list[int]],
    target_role: str | None,
    count: int,
) -> list[tuple[str, int]]:
    """Uniform sampling — one role or a diverse mix cycling through preferred roles."""
    if target_role:
        pool = index.get(target_role) or next(
            (v for k, v in index.items() if k.lower() == target_role.lower()), []
        )
        if not pool:
            print(f"ERROR: Role '{target_role}' not found. Use --list-roles.")
            sys.exit(1)
        return [(target_role, i) for i in pool[:count]]

    available: list[str] = []
    for r in _PREFERRED_ROLES:
        if r in index:
            available.append(r)
        else:
            m = next((k for k in index if k.lower() == r.lower()), None)
            if m:
                available.append(m)
    for r in index:
        if r not in available:
            available.append(r)

    result: list[tuple[str, int]] = []
    cursors: dict[str, int] = {r: 0 for r in available}
    while len(result) < count:
        added = 0
        for role in available:
            if len(result) >= count:
                break
            pool = index.get(role, [])
            cur = cursors[role]
            if cur < len(pool):
                result.append((role, pool[cur]))
                cursors[role] += 1
                added += 1
        if added == 0:
            break
    return result


def _pick_rows_mix(
    index: dict[str, list[int]],
    role_mix: str,
) -> list[tuple[str, int]]:
    """Parse 'RoleA:N,RoleB:M' and return that many rows per role."""
    result: list[tuple[str, int]] = []
    for segment in role_mix.split(","):
        segment = segment.strip()
        if ":" not in segment:
            print(f"ERROR: --role-mix segment '{segment}' must be in 'Role:count' format.")
            sys.exit(1)
        raw_role, raw_count = segment.rsplit(":", 1)
        raw_role = raw_role.strip()
        try:
            n = int(raw_count.strip())
        except ValueError:
            print(f"ERROR: count in '{segment}' is not an integer.")
            sys.exit(1)

        pool = index.get(raw_role) or next(
            (v for k, v in index.items() if k.lower() == raw_role.lower()), []
        )
        resolved_role = next(
            (k for k in index if k.lower() == raw_role.lower()), raw_role
        )
        if not pool:
            print(f"WARNING: Role '{raw_role}' not found in dataset — skipping.")
            continue
        for row_i in pool[:n]:
            result.append((resolved_role, row_i))
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download sample resumes + JD from HuggingFace as PDFs or text."
    )
    parser.add_argument("--role", default=None, help="Pull resumes from a single role.")
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT,
        help="Number of resumes (default 50). Ignored when --role-mix is set.",
    )
    parser.add_argument(
        "--role-mix", default=None,
        metavar="ROLE:N[,ROLE:N...]",
        help='E.g. "Software Engineer:5,Data Scientist:5,HR Manager:5"',
    )
    parser.add_argument(
        "--dest", default=None,
        help="Output directory (default: ~/Downloads/auto-resume-sifter-demo).",
    )
    parser.add_argument(
        "--format", default="pdf", choices=["pdf", "txt"],
        help="Output format: pdf (default) or txt.",
    )
    parser.add_argument(
        "--list-roles", action="store_true", help="Print available roles and exit.",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.dest) if args.dest else _DEFAULT_OUT_DIR
    fmt: str = args.format

    ds = _load_dataset()
    index = _build_index(ds)

    if args.list_roles:
        counts = {r: len(v) for r, v in index.items()}
        print(f"{'Role':<40} {'Count':>6}")
        print("-" * 48)
        for role, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"{role:<40} {cnt:>6}")
        return

    if args.role_mix:
        picks = _pick_rows_mix(index, args.role_mix)
    else:
        picks = _pick_rows_uniform(index, args.role, args.count)

    if not picks:
        print("ERROR: No resumes found.")
        sys.exit(1)

    # Remove existing files in destination so the folder is clean after re-run
    resume_dir = out_dir / "resumes" if not args.dest else out_dir
    resume_dir.mkdir(parents=True, exist_ok=True)
    for old in resume_dir.glob("resume_*.*"):
        old.unlink()

    print(f"Saving {len(picks)} resumes to: {resume_dir}\n")
    role_distribution: Counter = Counter()

    for seq, (role, row_i) in enumerate(picks, 1):
        row = ds[row_i]
        resume_text = str(row.get("Resume", "")).strip()
        if not resume_text:
            print(f"  (skipping row {row_i} — empty resume)")
            continue
        _save_resume(resume_text, role, seq, resume_dir, fmt)
        role_distribution[role] += 1

    print(f"\nRole breakdown:")
    for role, cnt in role_distribution.most_common():
        print(f"  {role}: {cnt}")

    # Save JD — prefer the most common role in the selection
    top_role = role_distribution.most_common(1)[0][0]
    jd_text = ""
    for row_i in index.get(top_role, []):
        row = ds[row_i]
        candidate_jd = str(row.get("Job_Description", "")).strip()
        if len(candidate_jd) > 200:
            jd_text = candidate_jd
            break

    if not jd_text:
        for row in ds:
            candidate_jd = str(row.get("Job_Description", "")).strip()
            if len(candidate_jd) > 200:
                jd_text = candidate_jd
                break

    if jd_text:
        ext = "pdf" if fmt == "pdf" else "txt"
        jd_path = out_dir / f"jd.{ext}"
        if fmt == "pdf":
            _render_pdf(jd_text, jd_path)
        else:
            jd_path.write_text(_wrap(jd_text), encoding="utf-8")
        # Remove old format if it exists
        other_ext = "txt" if fmt == "pdf" else "pdf"
        old_jd = out_dir / f"jd.{other_ext}"
        if old_jd.exists():
            old_jd.unlink()
        print(f"\nJob description saved to: {jd_path}  (Role: {top_role})")
    else:
        print("\nWARNING: No job description found in dataset.")

    resumes_path = resume_dir
    jd_ref = out_dir / f"jd.{'pdf' if fmt == 'pdf' else 'txt'}"
    print(f"\nDone. In the app, use 'Load from folder' and point to:")
    print(f"  Resumes : {resumes_path}")
    print(f"  JD      : {jd_ref}")


if __name__ == "__main__":
    main()
