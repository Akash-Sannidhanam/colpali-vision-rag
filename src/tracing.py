"""LangSmith tracing: the one module that imports `langsmith`.

LangGraph traces the four graph nodes on its own once `LANGSMITH_TRACING=true` and a
key are set. That leaves each node opaque, because `google-genai` is not a LangChain
model and nothing traces it. This module supplies the pieces that open the nodes up:
`traceable` for the Gemini choke point and the retrieval seam, the hooks that shape
what those runs record, and the trace URL the UI links to.

Two rules hold here:

- **Page images never leave the box.** `redact_contents` swaps every inline image for
  a size descriptor before the run is recorded. The `PAGE i (pdf p.N):` label that
  precedes each image is text and stays, so a trace still says which page was sent.
- **Off costs nothing and never fails a query.** `traceable` is a pass-through when
  tracing is disabled, and `trace_url` returns None on any failure rather than raising.

Kept free of `google.genai` imports (parts are duck-typed) so `src.retrieval`, which
must run with no API key for `--retrieval-only`, can depend on it.
"""

import time
import uuid
from typing import Any

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree, get_tracing_context
from langsmith.run_trees import get_cached_client
from langsmith.utils import get_tracer_project, tracing_is_enabled

from src import config
from src.logging_setup import get_logger

log = get_logger("tracing")

__all__ = [
    "annotate_current_run",
    "enabled",
    "gemini_outputs",
    "redact_contents",
    "retriever_outputs",
    "run_metadata",
    "trace_url",
    "traceable",
]

# How long a failed project lookup is remembered before `trace_url` tries again. The
# project is created by the first trace that lands in it, so a lookup on the very first
# query of a fresh project can fail legitimately; retrying on every query would put a
# network round-trip on the query path for as long as LangSmith is unreachable.
_URL_RETRY_S = 60.0
_project_url: str | None = None
_project_url_failed_at: float | None = None


def enabled() -> bool:
    """True when LangSmith tracing is on for this process (env or tracing context)."""
    return bool(tracing_is_enabled())


def _describe_part(part: Any) -> Any:
    """A Gemini content item as it should appear in a trace: images become descriptors."""
    if isinstance(part, str):
        return part
    inline = getattr(part, "inline_data", None)
    if inline is not None:
        size_kb = round(len(getattr(inline, "data", b"") or b"") / 1024)
        return f"<{getattr(inline, 'mime_type', 'image')} {size_kb}KB>"
    text = getattr(part, "text", None)
    return text if text is not None else repr(part)


def redact_contents(inputs: dict) -> dict:
    """`process_inputs` hook for `gemini_client.generate`: no image bytes in the trace."""
    out = dict(inputs)
    if isinstance(out.get("contents"), list):
        out["contents"] = [_describe_part(p) for p in out["contents"]]
    schema = out.get("response_schema")
    if schema is not None:
        out["response_schema"] = getattr(schema, "__name__", repr(schema))
    return out


def gemini_outputs(response: Any) -> dict:
    """`process_outputs` hook for `generate`: the parsed answer plus token usage.

    `usage_metadata` uses LangSmith's own field names, which is what makes the run show
    token counts and (with `ls_model_name` in metadata) a cost.
    """
    parsed = getattr(response, "parsed", None)
    if parsed is not None and hasattr(parsed, "model_dump"):
        body: Any = parsed.model_dump(mode="json")
    else:
        body = getattr(response, "text", None)
    out: dict = {"output": body}
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        out["usage_metadata"] = {
            "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(usage, "total_token_count", 0) or 0,
        }
    return out


def retriever_outputs(hits: list[dict]) -> dict:
    """`process_outputs` hook for `retrieval.retrieve`: hits in LangSmith's document shape,
    which its UI renders as a ranked list rather than a JSON blob."""
    return {"documents": [
        {
            "page_content": f"{h.get('pdf')} p.{h.get('page_number')}",
            "type": "Document",
            "metadata": {"pdf": h.get("pdf"), "page_number": h.get("page_number"),
                         "score": h.get("score")},
        }
        for h in hits
    ]}


def annotate_current_run(**metadata: Any) -> None:
    """Add metadata to the run in progress. A no-op when nothing is being traced."""
    run = get_current_run_tree()
    if run is not None:
        run.metadata.update(metadata)


def run_metadata() -> dict:
    """The retrieval/model knobs a trace should be filterable by - the same ones an eval
    report's config snapshot records, so a trace can be matched to its experiment arm."""
    return {
        "retrieve_k": config.RETRIEVE_K,
        "rerank_k": config.RERANK_K,
        "max_pages_per_doc": config.MAX_PAGES_PER_DOC,
        "query_decompose": config.QUERY_DECOMPOSE,
        "gemini_model": config.GEMINI_MODEL,
        "rerank_model": config.RERANK_MODEL,
        "embed_version": config.EMBED_VERSION,
        "collection": config.COLLECTION_NAME,
    }


def _resolve_project_url() -> str | None:
    """The project's LangSmith URL, looked up once and cached; None while unavailable."""
    global _project_url, _project_url_failed_at
    if _project_url is not None:
        return _project_url
    if _project_url_failed_at is not None and time.monotonic() - _project_url_failed_at < _URL_RETRY_S:
        return None
    try:
        # The tracing context's client when one is set (tests, a scoped context), else
        # the process-wide one the traces themselves are sent with.
        client = get_tracing_context()["client"] or get_cached_client()
        _project_url = client.read_project(project_name=get_tracer_project()).url
        _project_url_failed_at = None
    except Exception:
        _project_url_failed_at = time.monotonic()
        log.warning("could not resolve the LangSmith project URL; trace links omitted",
                    exc_info=True)
    return _project_url


def trace_url(run_id: str) -> str | None:
    """Link to the trace for `run_id`, or None when tracing is off or LangSmith is
    unreachable. Never raises: a missing link must not fail the query it describes."""
    if not enabled():
        return None
    project_url = _resolve_project_url()
    if not project_url:
        return None
    # The request id is uuid4().hex; LangSmith's URLs use the dashed form.
    return f"{project_url}/r/{uuid.UUID(run_id)}?poll=true"
