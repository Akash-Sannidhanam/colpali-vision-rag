"""Audit the cross-document eval rows for questions one page can answer by itself.

A cross-document row exists to measure whether retrieval reached **both** gold documents.
That premise breaks in two ways, and neither is visible from the metrics:

  A. **One page states both halves.** The question cannot test cross-document retrieval
     at all. `gold_doc_coverage` still scores it 1.0 whenever retrieval happens to reach
     both documents, so a defective row can sit at a perfect score indefinitely.
  B. **A half is answerable from a document outside its gold.** The question needs two
     documents, but not the two labelled - so a "miss" may be the pipeline reading the
     fact from a perfectly good page nobody labelled.

Both were found in the shipped dataset: 6 of 20 cross-document rows, one of which was
scoring 1.0. See the label-audit pass in docs/ENGINEERING_LOG.md.

This searches the text layer the retriever deliberately never reads, exactly like
`find_in_pdfs.py`, and it is a **labelling aid, not a test** - it needs the fetched
distractor corpus (`scripts/fetch_eval_corpus.py`), which CI does not have.

    uv run python scripts/audit_xdoc_labels.py                 # audit every cross-doc row
    uv run python scripts/audit_xdoc_labels.py --phrase A --phrase B   # vet a candidate pair

Exit codes: 0 = no defect found, 1 = at least one row defective, 2 = pdftotext missing.

**Only class A is decided automatically.** Two pages containing two phrases is a fact;
whether another document *answers a half* is a judgement a substring cannot make. So
class B is reported as REVIEW and never fails the run - "nDCG@10" appears in four papers
because four papers report nDCG@10, not because any of them states BEIR's choice of
cutoff.

**Read INCONCLUSIVE as "audit this by hand", not as a pass.** A phrase only locates a page
if it is specific: a bare number ("6", "18", "50") or a word spread across most of the
corpus ("English", "sharing") matches everywhere, which makes the co-occurrence test
vacuous rather than passing. Rows left with fewer than two specific phrases are reported,
not guessed at.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDFS_DIR = ROOT / "pdfs"
DATASET = ROOT / "eval" / "dataset.jsonl"

# A bare 1-3 digit label ("6", "18", "128") appears on hundreds of pages, so it cannot
# discriminate a page and its presence makes the whole-page test meaningless rather than
# passing. Rows left with fewer than two usable phrases are reported INCONCLUSIVE.
_BARE_NUMBER = re.compile(r"^\d{1,3}$")

# ...and neither can a word spread across most of the corpus. "English" is in 11 of the 19
# PDFs; co-occurrence with it says nothing. Three is the cut because a fact genuinely
# belonging to one paper is routinely echoed by one or two that cite it, and those are the
# cases class B exists to *flag* rather than the noise it has to ignore.
_MAX_DOCS_TO_DISCRIMINATE = 3


def page_texts(pdf: Path) -> list[str]:
    """Lower-cased text of each page, via `pdftotext -layout` (empty list if it fails)."""
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError):
        print(f"warning: could not read {pdf.name}", file=sys.stderr)
        return []
    return [page.lower() for page in out.split("\f")]


def load_corpus(pdfs_dir: Path) -> dict[str, list[str]]:
    return {pdf.name: page_texts(pdf) for pdf in sorted(pdfs_dir.glob("*.pdf"))}


def pages_with(corpus: dict[str, list[str]], phrase: str) -> list[str]:
    """Every "<pdf> p<n>" whose text contains `phrase`."""
    needle = phrase.lower()
    return [f"{name} p{n}" for name, pages in corpus.items()
            for n, text in enumerate(pages, start=1) if needle in text]


def documents_with(corpus: dict[str, list[str]], phrase: str) -> list[str]:
    """Every pdf containing `phrase` anywhere - the class-B check."""
    needle = phrase.lower()
    return sorted(name for name, pages in corpus.items() if any(needle in t for t in pages))


def audit_phrases(corpus: dict[str, list[str]], phrases: list[str]) -> dict:
    """Both checks for one set of discriminating phrases, one per gold document."""
    shared = [p for p in pages_with(corpus, phrases[0])
              if all(ph.lower() in corpus[p.split()[0]][int(p.split()[1][1:]) - 1]
                     for ph in phrases[1:])]
    spread = {ph: documents_with(corpus, ph) for ph in phrases}
    return {
        "shared_pages": shared,                                        # class A
        "multi_document": {k: v for k, v in spread.items() if len(v) > 1},  # class B
    }


def cross_doc_rows(dataset_path: Path) -> list[dict]:
    rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    return [r for r in rows if len({g["pdf"] for g in r["gold"]}) > 1]


def discriminating_phrases(row: dict, corpus: dict[str, list[str]]) -> list[str]:
    """The row's own reference facts, minus the ones too generic to locate a page."""
    return [
        x for x in (row.get("answer_contains_all") or [])
        if not _BARE_NUMBER.match(x)
        and 0 < len(documents_with(corpus, x)) <= _MAX_DOCS_TO_DISCRIMINATE
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--dir", default=str(PDFS_DIR), metavar="DIR")
    parser.add_argument("--phrase", action="append", metavar="TEXT",
                        help="vet a candidate phrase set instead of the dataset "
                             "(repeat once per gold document)")
    args = parser.parse_args(argv)

    if not shutil.which("pdftotext"):
        print("pdftotext not found on PATH - install poppler", file=sys.stderr)
        return 2
    corpus = load_corpus(Path(args.dir))
    if not corpus:
        print(f"no PDFs under {args.dir} - run scripts/fetch_eval_corpus.py", file=sys.stderr)
        return 2

    # --phrase mode: vet a pair before writing it into the dataset.
    if args.phrase:
        if len(args.phrase) < 2:
            parser.error("--phrase needs at least two phrases, one per gold document")
        result = audit_phrases(corpus, args.phrase)
        for phrase in args.phrase:
            docs = documents_with(corpus, phrase)
            print(f"  {phrase!r:38s} -> {docs or ['(absent)']}")
        if result["shared_pages"]:
            print(f"\nDEFECT (A): one page has all of them: {result['shared_pages'][:5]}")
        if result["multi_document"]:
            print(f"DEFECT (B): not single-document: {result['multi_document']}")
        ok = not result["shared_pages"] and not result["multi_document"]
        print("\nusable as a cross-document pair" if ok else "\nnot usable as-is")
        return 0 if ok else 1

    defective = inconclusive = review = 0
    for row in cross_doc_rows(Path(args.dataset)):
        phrases = discriminating_phrases(row, corpus)
        if len(phrases) < 2:
            inconclusive += 1
            print(f"INCONCLUSIVE  {row['id']:44s} <2 phrases specific enough to locate a page")
            continue
        result = audit_phrases(corpus, phrases)
        if result["shared_pages"]:
            defective += 1
            print(f"DEFECT (A)    {row['id']:44s} one page states both: "
                  f"{', '.join(result['shared_pages'][:3])}")
        elif result["multi_document"]:
            review += 1
            print(f"REVIEW (B)    {row['id']:44s} also appears elsewhere: "
                  f"{result['multi_document']}")
        else:
            print(f"ok            {row['id']:44s} {phrases}")

    print(f"\n{defective} defective, {review} to review, {inconclusive} inconclusive "
          f"(review + inconclusive need a human)")
    return 1 if defective else 0


if __name__ == "__main__":
    sys.exit(main())
