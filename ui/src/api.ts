import type {
  CorpusResponse,
  DeleteResponse,
  DocumentPagesResponse,
  HealthResponse,
  HeatmapResponse,
  ImageRef,
  QueryResponse,
} from './types'

// The backend builds absolute image URLs from its own base_url, so image srcs are
// usually already absolute; BASE is the fallback for relative paths and the fetch root.
//
// A production build defaults to '' — relative, same-origin — because the deployed
// shape is FastAPI serving this bundle itself (see the Dockerfile). Only `vite dev`
// needs an absolute target, since it runs on :5173 while the API is on :8000.
// VITE_API_BASE overrides both, for pointing a dev UI at a remote backend.
const BASE = (
  import.meta.env.VITE_API_BASE ?? (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '')
).replace(/\/$/, '')

/** Resolve an ImageRef to an <img src>: prefer the inline data-URI, else the URL. */
export function imageSrc(ref: ImageRef | null | undefined): string | undefined {
  if (!ref) return undefined
  if (ref.data_uri) return ref.data_uri
  if (ref.url) return ref.url.startsWith('http') ? ref.url : `${BASE}${ref.url}`
  return undefined
}

// --- API key ---
//
// Held in sessionStorage, never baked into the bundle: this is static JS shipped to the
// browser, so a build-time key would be readable by anyone who loads the page. The
// operator pastes the key once per tab session.
//
// Image requests are deliberately NOT authenticated — <img src> cannot send a custom
// header, which is why the backend leaves /images open (see src/auth.py).
const KEY_STORAGE = 'visionrag.apiKey'

/** The stored API key, or '' when none has been entered. */
export function getApiKey(): string {
  return sessionStorage.getItem(KEY_STORAGE) ?? ''
}

/** Remember the key for this tab session. */
export function setApiKey(key: string): void {
  sessionStorage.setItem(KEY_STORAGE, key)
}

/** Forget the key — called automatically when the server rejects it. */
export function clearApiKey(): void {
  sessionStorage.removeItem(KEY_STORAGE)
}

/** Thrown on a 401 so the UI can prompt for a key instead of showing a raw error. */
export class UnauthorizedError extends Error {
  constructor(message = 'This server requires an API key.') {
    super(message)
    this.name = 'UnauthorizedError'
  }
}

/** Thrown on a 429 so the UI can say "slow down" rather than "request failed". */
export class RateLimitedError extends Error {
  constructor(
    message: string,
    readonly retryAfterSeconds: number | null,
  ) {
    super(message)
    this.name = 'RateLimitedError'
  }
}

/**
 * fetch against the API base, carrying the stored key when there is one.
 *
 * Every call goes through here, so the header is attached in exactly one place and a
 * new endpoint cannot forget it.
 */
function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const key = getApiKey()
  const headers = new Headers(init.headers)
  if (key) headers.set('X-API-Key', key)
  return fetch(`${BASE}${path}`, { ...init, headers })
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* keep statusText */
    }
    if (res.status === 401) {
      // The stored key is wrong or the server started requiring one; drop it so the
      // next render prompts rather than retrying a key we know is rejected.
      clearApiKey()
      throw new UnauthorizedError(detail)
    }
    if (res.status === 429) {
      const retry = Number(res.headers.get('Retry-After'))
      throw new RateLimitedError(detail, Number.isFinite(retry) && retry > 0 ? retry : null)
    }
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json() as Promise<T>
}

export async function query(
  question: string,
  opts: { inline?: boolean } = {},
): Promise<QueryResponse> {
  const suffix = opts.inline ? '?inline=true' : ''
  const res = await apiFetch(`/query${suffix}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  return asJson<QueryResponse>(res)
}

/** Per-patch MaxSim heatmap for the given page ("why this page?"). On-demand: called
 *  only when the Viewer toggle is enabled, since it costs extra model forward passes. */
export async function heatmap(
  question: string,
  pdf: string,
  pageNumber: number,
): Promise<HeatmapResponse> {
  const res = await apiFetch('/heatmap', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, pdf, page_number: pageNumber }),
  })
  return asJson<HeatmapResponse>(res)
}

/** /health returns 503 (not throw) when degraded; parse the body either way. It is the
 *  one endpoint the backend leaves ungated, so it answers before a key is entered. */
export async function getHealth(): Promise<HealthResponse> {
  const res = await apiFetch('/health')
  return res.json() as Promise<HealthResponse>
}

export async function getCorpus(): Promise<CorpusResponse> {
  return asJson<CorpusResponse>(await apiFetch('/corpus'))
}

/** Every page of one document, with its /images URL - the full-screen viewer's manifest. */
export async function getDocumentPages(pdf: string): Promise<DocumentPagesResponse> {
  const res = await apiFetch(`/corpus/${encodeURIComponent(pdf)}/pages`)
  return asJson<DocumentPagesResponse>(res)
}

/**
 * A document's original PDF, as bytes.
 *
 * The endpoint is gated, and `<a href download>` cannot send X-API-Key any more than
 * `<img src>` can - the same limitation that keeps /images open. It is gated anyway
 * (/images exposes derived page rasters; this hands over whole source documents), so the
 * bytes come through fetch, and what the caller does with them is its business: the
 * download button wraps them in an object URL, the viewer hands them to pdf.js.
 *
 * A Blob rather than an ArrayBuffer on purpose. `pdfjs.getDocument({data})` *detaches*
 * the buffer it is given, so a viewer that also wants to offer a download must be able to
 * produce the bytes twice - and `Blob.arrayBuffer()` returns a fresh copy every call.
 */
export async function getDocumentFile(pdf: string): Promise<Blob> {
  const res = await apiFetch(`/corpus/${encodeURIComponent(pdf)}/file`)
  // Route failures through asJson so a 401/429 raises the same typed error the JSON
  // endpoints do. asJson always throws for a non-2xx, so control never returns here.
  if (!res.ok) await asJson<never>(res)
  return res.blob()
}

/** Hand a blob to the browser as a download, via a short-lived object URL. */
export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  // Revoked on the next task rather than synchronously: Safari cancels a download whose
  // object URL is revoked in the same tick as the click.
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

/** Remove a document from the corpus: vectors, page images, crops, and the stored PDF. */
export async function deleteDocument(pdf: string): Promise<DeleteResponse> {
  const res = await apiFetch(`/corpus/${encodeURIComponent(pdf)}`, { method: 'DELETE' })
  return asJson<DeleteResponse>(res)
}

// One progress event from POST /ingest/stream (mirrors src.ingest event dicts).
export interface IngestEvent {
  phase: 'render' | 'pages' | 'embed' | 'stored' | 'skip' | 'done' | 'error'
  pdf?: string
  page?: number
  total?: number
  count?: number
  indexed_pages?: number
  detail?: string
}

/**
 * Upload a PDF and stream per-page ingest progress, calling `onEvent` for each event.
 * Uses fetch + a ReadableStream reader (EventSource can't POST multipart) to parse the
 * SSE `data: {json}\n\n` frames. Rejects on a non-2xx (the endpoint validates the
 * upload before it starts streaming) or on a terminal `error` event.
 */
export async function ingestStream(
  file: File,
  onEvent: (e: IngestEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const fd = new FormData()
  fd.append('file', file)
  const res = await apiFetch('/ingest/stream', { method: 'POST', body: fd, signal })
  // Route failures through asJson so a 401/429 raises the same typed error the JSON
  // endpoints do (the endpoint validates the upload before it starts streaming).
  // asJson always throws for a non-2xx, so control never returns from this branch.
  if (!res.ok) await asJson<never>(res)
  if (!res.body) throw new Error('The server returned no response body.')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let sep: number
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const line = frame.split('\n').find((l) => l.startsWith('data:'))
      if (line) onEvent(JSON.parse(line.slice(5).trim()) as IngestEvent)
    }
  }
}
