"""Fetch the eval distractor corpus listed in eval/corpus_manifest.json into pdfs/.

The eval measures nothing useful on a tiny corpus: with RETRIEVE_K=10 over the three
demo documents (43 pages), every query pulls back ~23% of the whole index, so recall@10
is pinned at 1.0 and every downstream metric with it. These papers are the haystack -
chosen to be *topically confusable* with the gold documents (transformer architectures,
document-AI vision models, and late-interaction retrievers) rather than filler.

The PDFs themselves are deliberately not committed; the manifest pins each one by
sha256 so the corpus is reproducible without adding ~200 MB to the repo.

    uv run python scripts/fetch_eval_corpus.py                  # fetch what's missing
    uv run python scripts/fetch_eval_corpus.py --verify         # check, download nothing
    uv run python scripts/fetch_eval_corpus.py --update-hashes  # adopt current bytes

Self-contained on purpose (no `src.` imports), so it runs without PYTHONPATH=. like
scripts/make_sample_pdf.py. Poppler must be on PATH for the page count, same as ingest.

Exit codes: 0 = corpus complete; 1 = something is missing, corrupt, or unreachable.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from pdf2image import pdfinfo_from_path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "eval" / "corpus_manifest.json"
PDFS_DIR = ROOT / "pdfs"

# arXiv asks automated clients to identify themselves and to space out requests.
_USER_AGENT = "colpali-vision-rag-eval-corpus/1.0 (+https://github.com/Akash-Sannidhanam)"
_REQUEST_SPACING_S = 3.0
_TIMEOUT_S = 120.0


class FetchError(Exception):
    """Anything that leaves the corpus incomplete (exit code 1)."""


def load_manifest(path: Path) -> list[dict]:
    """Parse and validate the manifest, raising FetchError naming the offending entry."""
    try:
        entries = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FetchError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(entries, list) or not entries:
        raise FetchError(f"manifest {path} must be a non-empty JSON list")

    seen: set[str] = set()
    for i, entry in enumerate(entries):
        label = entry.get("name", f"entry {i}") if isinstance(entry, dict) else f"entry {i}"
        if not isinstance(entry, dict):
            raise FetchError(f"{label}: each manifest entry must be a JSON object")
        name = entry.get("name")
        # A name with a path separator would escape pdfs/ entirely; the manifest is
        # tracked, but treat it as untrusted input anyway - it names a write target.
        if not isinstance(name, str) or not name.endswith(".pdf") or Path(name).name != name:
            raise FetchError(f"{label}: `name` must be a bare *.pdf filename")
        if name in seen:
            raise FetchError(f"{label}: duplicate name")
        seen.add(name)
        if not isinstance(entry.get("url"), str) or not entry["url"].startswith("https://"):
            raise FetchError(f"{label}: `url` must be an https URL")
    return entries


def sha256_of(path: Path) -> str:
    """Hash a file in chunks, so a big PDF isn't slurped into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def page_count(path: Path) -> int:
    """Page count via poppler; 0 when it can't be read (informational field only)."""
    try:
        return int(pdfinfo_from_path(str(path), poppler_path=os.getenv("POPPLER_PATH"))["Pages"])
    except Exception:
        return 0


def download(url: str, dest: Path) -> None:
    """Download `url` to `dest`, writing through a sibling temp file.

    The partial download lands on `<dest>.part`, never on a `*.pdf` path, because
    `ingest.py` globs `pdfs/*.pdf` - an interrupted fetch must not leave a truncated
    file where the next ingest would happily try to embed it.
    """
    partial = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response, \
                partial.open("wb") as out:
            while chunk := response.read(1024 * 256):
                out.write(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        partial.unlink(missing_ok=True)
        raise FetchError(f"download failed for {url}: {exc}") from exc
    partial.replace(dest)


def _state(entry: dict, path: Path) -> str:
    """Classify one entry's local file: missing / stale / unpinned / ok."""
    if not path.exists():
        return "missing"
    expected = entry.get("sha256")
    if not expected:
        return "unpinned"
    return "ok" if sha256_of(path) == expected else "stale"


def fetch_all(entries: list[dict], *, verify_only: bool, update_hashes: bool) -> int:
    """Bring pdfs/ in line with the manifest; return the number of problems found."""
    problems = 0
    downloaded = 0
    for entry in entries:
        name, path = entry["name"], PDFS_DIR / entry["name"]
        state = _state(entry, path)

        if state == "ok":
            print(f"ok       {name}")
            continue
        if state == "stale" and not update_hashes:
            # arXiv can serve a revised version of a paper under the same URL. Failing
            # loudly matters: a silently-changed corpus invalidates every stored report,
            # because the whole point of the baseline is that the haystack is fixed.
            print(f"MISMATCH {name}: sha256 differs from the manifest "
                  f"(re-pin with --update-hashes if the change is intended)", file=sys.stderr)
            problems += 1
            continue
        if verify_only:
            print(f"MISSING  {name}", file=sys.stderr)
            problems += 1
            continue

        if state == "missing":
            if downloaded:
                time.sleep(_REQUEST_SPACING_S)
            print(f"fetch    {name}  <- {entry['url']}")
            download(entry["url"], path)
            downloaded += 1
            if entry.get("sha256") and not update_hashes:
                actual = sha256_of(path)
                if actual != entry["sha256"]:
                    path.unlink(missing_ok=True)
                    raise FetchError(
                        f"{name}: sha256 {actual} does not match the manifest "
                        f"{entry['sha256']} - the URL served different bytes"
                    )

        if update_hashes:
            entry["sha256"] = sha256_of(path)
            entry["pages"] = page_count(path)
            print(f"pinned   {name}  {entry['sha256'][:12]}...  {entry['pages']} pages")

    return problems


def main(argv: list[str] | None = None) -> int:
    """Parse args, reconcile pdfs/ with the manifest, return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--verify", action="store_true",
                        help="report the corpus state without downloading anything")
    parser.add_argument("--update-hashes", action="store_true",
                        help="record the sha256/pages of the files on disk into the manifest "
                             "(use when adding an entry, or to adopt an intended revision)")
    args = parser.parse_args(argv)
    if args.verify and args.update_hashes:
        parser.error("--verify downloads nothing; drop it to use --update-hashes")

    manifest_path = Path(args.manifest)
    try:
        entries = load_manifest(manifest_path)
        PDFS_DIR.mkdir(parents=True, exist_ok=True)
        problems = fetch_all(entries, verify_only=args.verify, update_hashes=args.update_hashes)
    except FetchError as exc:
        print(f"eval corpus error: {exc}", file=sys.stderr)
        return 1

    if args.update_hashes:
        manifest_path.write_text(json.dumps(entries, indent=2) + "\n")
        print(f"\nmanifest updated: {manifest_path}")

    total_pages = sum(e.get("pages", 0) for e in entries)
    print(f"\n{len(entries)} documents, {total_pages} pages in {PDFS_DIR}")
    if problems:
        print(f"{problems} problem(s); re-run without --verify to fetch", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
