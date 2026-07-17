# Enhancements / Backlog

Possible improvements to the vision-citation pipeline. None are blocking: the
core feature (Gemini returns a bounding region, which is cropped and shown to
the reader) is complete, tested, and shipped. Captured here so they are not lost.

## Display

- **Inline UI for the crop.** Today the answer prints to the terminal and the
  crop opens in macOS Preview (`_open_file` in `src/main.py`). A small Streamlit
  or Gradio app would render the answer, the cropped slice, and the annotated
  page together in the browser, which is the natural home for a "show the reader
  the exact slice" feature. Highest-value next step.
- **Cross-platform auto-open.** `_open_file` in `src/main.py` only handles macOS
  (`open`). Add `xdg-open` (Linux) and `os.startfile` / `start` (Windows), or
  drop auto-open entirely once an inline UI exists.

## Artifacts

- **Persist crops per query.** `crop_region` and `annotate_page` in
  `src/highlight.py` write `<page_stem>_crop.png` and overwrite on every run.
  Adding a short hash of the question to the filename keeps a history across
  queries instead of clobbering the previous one.

## Robustness

- **Harden structured-output parsing.** `src/answerer.py` falls back to
  `json.loads(response.text)` when `response.parsed` is None; a malformed
  response would raise. Catch that and return a not-found result so the CLI
  degrades gracefully.
- **Confirm the Gemini model id.** `GEMINI_MODEL = "gemini-3.5-flash"` in
  `src/config.py` works in testing but is an unusual string. Pin or verify the
  intended model.

## Scope

- **Multiple regions.** The pipeline cites a single primary region (a deliberate
  choice). If an answer ever spans two pages or two areas, extend the `Citation`
  schema in `src/answerer.py` to a list of boxes and crop each.
- **Integration test with a mocked Gemini.** Current tests cover the pure
  geometry in `src/highlight.py`. A test that stubs the Gemini call would cover
  the `answer_node` to `highlight_node` wiring without needing an API key.
