import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  RateLimitedError,
  UnauthorizedError,
  clearApiKey,
  getApiKey,
  getCorpus,
  imageSrc,
  ingestStream,
  setApiKey,
} from './api'
import type { IngestEvent } from './api'

/**
 * `api.ts` under vitest - the parsing layer, with fetch stubbed.
 *
 * These are node tests for the same reason lib.test.ts is: nothing here is layout. What
 * they cover is the part of the client with real branches and no other coverage - the
 * status-to-typed-error mapping every component's catch depends on, and the SSE frame
 * parser, which the browser suite cannot reach at the level that matters (it can drive an
 * ingest, but not one whose frames arrive split across chunk boundaries).
 */

// sessionStorage is not in node. A Map-backed stand-in is enough: api.ts only ever does
// getItem/setItem/removeItem on one key.
function stubSessionStorage() {
  const store = new Map<string, string>()
  vi.stubGlobal('sessionStorage', {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
    removeItem: (k: string) => void store.delete(k),
  })
}

/** A Response whose body is `detail` as JSON, at `status`. */
function errorResponse(status: number, detail: string, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify({ detail }), { status, headers })
}

beforeEach(() => {
  stubSessionStorage()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('imageSrc', () => {
  it('prefers the inline data-URI over the URL', () => {
    // The backend sends both when ?inline=true. The data-URI is already in hand, so
    // using the URL would be a second round trip for bytes we have.
    expect(imageSrc({ url: '/images/p1.png', data_uri: 'data:image/png;base64,AAA' })).toBe(
      'data:image/png;base64,AAA',
    )
  })

  it('passes an absolute URL through untouched', () => {
    // The server builds these from its own base_url, so prefixing BASE would corrupt them.
    expect(imageSrc({ url: 'http://elsewhere/images/p1.png', data_uri: null })).toBe(
      'http://elsewhere/images/p1.png',
    )
  })

  it('resolves a relative URL against the API base', () => {
    // Under vitest `import.meta.env.DEV` is true, so BASE is the dev default - the same
    // branch `vite dev` takes, where the UI is on :5173 and the API on :8000. A
    // production build resolves BASE to '' and the same input stays relative; what is
    // under test either way is that a relative URL gets the base prepended and an
    // absolute one does not.
    expect(imageSrc({ url: '/images/p1.png', data_uri: null })).toBe(
      'http://127.0.0.1:8000/images/p1.png',
    )
  })

  it('is undefined for a ref that carries neither, and for no ref', () => {
    // `image: null` is a real state: the page is indexed but its PNG is gone from disk.
    expect(imageSrc({ url: null, data_uri: null })).toBeUndefined()
    expect(imageSrc(null)).toBeUndefined()
    expect(imageSrc(undefined)).toBeUndefined()
  })
})

describe('the status-to-error mapping', () => {
  it('raises UnauthorizedError on 401 and drops the stored key', async () => {
    // Dropping it matters: without that the next render retries a key the server has
    // already rejected, instead of prompting for a new one.
    setApiKey('stale-key')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(errorResponse(401, 'Missing API key.')))

    await expect(getCorpus()).rejects.toBeInstanceOf(UnauthorizedError)
    expect(getApiKey()).toBe('')
  })

  it('raises RateLimitedError on 429, carrying Retry-After', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(errorResponse(429, 'Too many requests.', { 'Retry-After': '30' })),
    )

    await expect(getCorpus()).rejects.toMatchObject({
      name: 'RateLimitedError',
      retryAfterSeconds: 30,
    })
  })

  it('reports Retry-After as null when it is absent or unusable', async () => {
    // The UI words the message differently with and without a number, so "unknown" has to
    // be distinguishable from a real value - not silently coerced to 0, which would read
    // as "retry immediately" and walk straight back into the limit.
    const cases: Record<string, string>[] = [
      {},                                                   // absent
      { 'Retry-After': 'Wed, 21 Oct 2026 07:28:00 GMT' },   // the HTTP-date form, unparsed
      { 'Retry-After': '0' },                               // present but meaningless
    ]
    for (const headers of cases) {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(errorResponse(429, 'slow down', headers)))
      const err = await getCorpus().catch((e) => e)
      expect(err).toBeInstanceOf(RateLimitedError)
      expect(err.retryAfterSeconds).toBeNull()
    }
  })

  it('raises a plain Error carrying status and detail for anything else', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(errorResponse(503, 'Qdrant is unreachable.')))
    await expect(getCorpus()).rejects.toThrow('503: Qdrant is unreachable.')
  })

  it('falls back to statusText when the error body is not JSON', async () => {
    // FastAPI answers JSON, but a proxy or a gateway in front of it need not.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response('<html>502</html>', { status: 502, statusText: 'Bad Gateway' })),
    )
    await expect(getCorpus()).rejects.toThrow('502: Bad Gateway')
  })
})

describe('the API key header', () => {
  it('is attached when a key is stored', async () => {
    setApiKey('sekret')
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await getCorpus()
    const headers = fetchMock.mock.calls[0][1].headers as Headers
    expect(headers.get('X-API-Key')).toBe('sekret')
  })

  it('is omitted entirely when there is none', async () => {
    // Not sent as an empty string: an ungated server should see a request with no key at
    // all, and `X-API-Key: ''` is a different thing that a gated one would reject.
    clearApiKey()
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await getCorpus()
    expect((fetchMock.mock.calls[0][1].headers as Headers).has('X-API-Key')).toBe(false)
  })
})

describe('ingestStream', () => {
  /** A Response streaming `chunks` verbatim, one read() per chunk. */
  function sse(chunks: string[], status = 200): Response {
    const encoder = new TextEncoder()
    const body = new ReadableStream({
      start(controller) {
        for (const c of chunks) controller.enqueue(encoder.encode(c))
        controller.close()
      },
    })
    return new Response(body, { status, headers: { 'Content-Type': 'text/event-stream' } })
  }

  const file = () => new File([new Uint8Array([1, 2, 3])], 'doc.pdf', { type: 'application/pdf' })

  it('parses one event per frame', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        sse([
          'data: {"phase":"render","pdf":"doc.pdf"}\n\n',
          'data: {"phase":"embed","page":1,"total":3}\n\n',
          'data: {"phase":"done","pdf":"doc.pdf","indexed_pages":3}\n\n',
        ]),
      ),
    )

    const seen: IngestEvent[] = []
    await ingestStream(file(), (e) => seen.push(e))
    expect(seen.map((e) => e.phase)).toEqual(['render', 'embed', 'done'])
    expect(seen[2].indexed_pages).toBe(3)
  })

  it('reassembles a frame split across chunk boundaries', async () => {
    // The reason the parser buffers at all. Chunk boundaries are the network's business,
    // not the protocol's - a frame can be cut anywhere, including mid-JSON and between
    // the two newlines that terminate it. This is the case the browser suite cannot
    // stage, because it does not control how the response is chunked.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        sse(['data: {"phase":"emb', 'ed","page":2,"total":9}\n', '\ndata: {"phase":"done"}\n\n']),
      ),
    )

    const seen: IngestEvent[] = []
    await ingestStream(file(), (e) => seen.push(e))
    expect(seen).toEqual([{ phase: 'embed', page: 2, total: 9 }, { phase: 'done' }])
  })

  it('delivers several frames arriving in one chunk', async () => {
    // The other half of the same problem: reads are not aligned to frames in either
    // direction, so a slow consumer can see a whole batch at once.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(sse(['data: {"phase":"pages","total":2}\n\ndata: {"phase":"embed","page":1}\n\n'])),
    )

    const seen: IngestEvent[] = []
    await ingestStream(file(), (e) => seen.push(e))
    expect(seen.map((e) => e.phase)).toEqual(['pages', 'embed'])
  })

  it('raises the same typed errors the JSON endpoints do', async () => {
    // The endpoint validates the upload before it starts streaming, so a 401 here has to
    // reach the key prompt exactly as one from /query would.
    setApiKey('stale')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(errorResponse(401, 'Missing API key.')))
    await expect(ingestStream(file(), () => {})).rejects.toBeInstanceOf(UnauthorizedError)
  })

  it('rejects when the server sends a terminal error event', async () => {
    // With an inert callback, so this tests ingestStream's own contract rather than
    // IngestModal's. It used to pass only because that one component happens to throw
    // from inside its onEvent - the documented behaviour was not in the function.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(sse(['data: {"phase":"error","detail":"Not a PDF."}\n\n'])),
    )
    await expect(ingestStream(file(), () => {})).rejects.toThrow('Not a PDF.')
  })

  it('hands the error event to the caller before raising', async () => {
    // So a caller can render the server's own wording; the throw is the safety net, not
    // the notification.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(sse(['data: {"phase":"error","detail":"Not a PDF."}\n\n'])),
    )
    const seen: IngestEvent[] = []
    await ingestStream(file(), (e) => seen.push(e)).catch(() => {})
    expect(seen).toEqual([{ phase: 'error', detail: 'Not a PDF.' }])
  })
})
