"""Find which page of which PDF states a fact - a labeling aid for eval/dataset.jsonl.

**The RAG pipeline never reads a text layer.** Ingest renders every page to an image and
ColQwen2 embeds the pixels; that is the whole premise. This script is not part of it. It
exists only so that *writing* the eval labels is verifiable instead of eyeballed, by
searching the text layer the retriever deliberately ignores.

Two jobs, both of which caught real mistakes while building the de-saturated eval:

1. **Locate a gold page.** A `{pdf, page}` gold label that names only the first mention
   of a fact makes the retriever look wrong when it finds a later one. Three such rows
   shipped before this script existed, and one pushed its gold page out of the top 10
   entirely - reported as a retrieval failure when it was a labeling failure.

2. **Prove an `unanswerable` question really is unanswerable.** A negative row is only
   worth having if the fact is absent from the *whole* corpus, and "absent from the two
   documents I was thinking about" is not the same claim. Exit code 1 (no matches) is
   the answer you want here.

    uv run python scripts/find_in_pdfs.py "English.to.Russian"       # absent? (want exit 1)
    uv run python scripts/find_in_pdfs.py "N = 6" --pdf attention    # which page states it
    uv run python scripts/find_in_pdfs.py "nDCG@5" --pdf colpali     # every page, not just the first

Self-contained (no `src.` imports), so it runs without PYTHONPATH=. like the other
scripts. Needs poppler's `pdftotext` on PATH, same as ingest.

Exit codes follow grep: 0 = at least one match, 1 = none, 2 = pdftotext missing.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

PDFS_DIR = Path(__file__).resolve().parent.parent / "pdfs"
MAX_LINE = 160  # keep one hit to one terminal line; layout mode preserves wide tables


def pages_of(pdf: Path) -> list[str]:
    """Text of each page, in order, via `pdftotext -layout` (empty list if it fails).

    `-layout` keeps columns and table cells roughly where they sit on the page, which
    matters because most gold facts in this corpus live in result tables.
    """
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError):
        print(f"warning: could not read {pdf.name}", file=sys.stderr)
        return []
    return out.split("\f")  # pdftotext separates pages with a form feed


def search(rx: re.Pattern, only: str | None, pdfs_dir: Path) -> int:
    """Print every matching line as `<pdf> p.<n> <line>`; return the match count."""
    matches = 0
    for pdf in sorted(pdfs_dir.glob("*.pdf")):
        if only and only.lower() not in pdf.name.lower():
            continue
        for number, page in enumerate(pages_of(pdf), start=1):
            for line in page.splitlines():
                if rx.search(line):
                    print(f"{pdf.name:24s} p.{number:<4d} {line.strip()[:MAX_LINE]}")
                    matches += 1
    return matches


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the search, and return a grep-style exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="A pixel-only PDF (the generated sales report) has no text layer, so it "
               "never matches - check those against the page PNGs by eye.",
    )
    parser.add_argument("pattern", help="regular expression, matched case-insensitively")
    parser.add_argument("--pdf", default=None, metavar="SUBSTR",
                        help="only search PDFs whose filename contains SUBSTR")
    parser.add_argument("--dir", default=str(PDFS_DIR), metavar="DIR",
                        help=f"directory of PDFs to search (default {PDFS_DIR})")
    args = parser.parse_args(argv)

    if not shutil.which("pdftotext"):
        print("pdftotext not found on PATH - install poppler (brew install poppler / "
              "apt-get install poppler-utils)", file=sys.stderr)
        return 2
    try:
        rx = re.compile(args.pattern, re.IGNORECASE)
    except re.error as exc:
        parser.error(f"bad pattern {args.pattern!r}: {exc}")

    matches = search(rx, args.pdf, Path(args.dir))
    if not matches:
        # The useful signal when vetting an `unanswerable` row: nothing in the corpus
        # states this, so a question about it has no answer to find.
        print(f"no match for {args.pattern!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
