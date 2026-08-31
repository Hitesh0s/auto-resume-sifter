import pathlib
import pdfplumber


def extract_text(path: str) -> str:
    """Extract plain text from a PDF or TXT file."""
    p = pathlib.Path(path)
    suffix = p.suffix.lower()

    if suffix == ".txt":
        raw = p.read_text(encoding="utf-8", errors="replace")
        return _normalise(raw)

    if suffix == ".pdf":
        return _extract_pdf(path)

    raise ValueError(f"Unsupported file type: {suffix!r}. Only PDF and TXT are accepted.")


def _extract_pdf(path: str) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:
        raise ValueError(f"Could not open PDF '{path}': {exc}") from exc

    text = "\n".join(pages)
    if not text.strip():
        raise ValueError(
            f"No text could be extracted from '{pathlib.Path(path).name}'. "
            "The file may be a scanned image or encrypted. "
            "Please upload a text-based PDF."
        )
    return _normalise(text)


def _normalise(text: str) -> str:
    text = text.replace("\x00", "")
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)
