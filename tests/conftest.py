"""Suite-wide setup that has to happen before any `src` module is imported.

LangSmith tracing is forced off. `src.config` calls `load_dotenv()` at import, so a
developer `.env` with `LANGSMITH_TRACING=true` would otherwise make `uv run pytest`
upload a trace for every stubbed query. Setting it here, at conftest import, works
where a fixture would not: `load_dotenv` never overrides a variable that is already
set, and langsmith caches its env reads on first use.
"""

import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
