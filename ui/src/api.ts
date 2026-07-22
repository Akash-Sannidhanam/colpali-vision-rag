import type {
  CorpusResponse,
  HealthResponse,
  IngestResponse,
  ImageRef,
  QueryResponse,
} from './types'

// The backend builds absolute image URLs from its own base_url, so image srcs are
// usually already absolute; BASE is the fallback for relative paths and the fetch root.
const BASE = (import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000').replace(/\/$/, '')

/** Resolve an ImageRef to an <img src>: prefer the inline data-URI, else the URL. */
export function imageSrc(ref: ImageRef | null | undefined): string | undefined {
  if (!ref) return undefined
  if (ref.data_uri) return ref.data_uri
  if (ref.url) return ref.url.startsWith('http') ? ref.url : `${BASE}${ref.url}`
  return undefined
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* keep statusText */
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
  const res = await fetch(`${BASE}/query${suffix}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  return asJson<QueryResponse>(res)
}

/** /health returns 503 (not throw) when degraded; parse the body either way. */
export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`)
  return res.json() as Promise<HealthResponse>
}

export async function getCorpus(): Promise<CorpusResponse> {
  return asJson<CorpusResponse>(await fetch(`${BASE}/corpus`))
}

export async function ingest(file: File): Promise<IngestResponse> {
  const fd = new FormData()
  fd.append('file', file)
  return asJson<IngestResponse>(await fetch(`${BASE}/ingest`, { method: 'POST', body: fd }))
}

// One progress event from POST /ingest/stream (mirrors src.ingest event dicts).
export interface IngestEvent {
  phase: 'render' | 'pages' | 'embed' | 'stored' | 'done' | 'error'
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
  const res = await fetch(`${BASE}/ingest/stream`, { method: 'POST', body: fd, signal })
  if (!res.ok || !res.body) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* keep statusText */
    }
    throw new Error(`${res.status}: ${detail}`)
  }

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
