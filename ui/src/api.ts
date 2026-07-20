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
