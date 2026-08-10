import { useCallback, useEffect, useState } from 'react'
import { UnauthorizedError, deleteDocument, getCorpus, getHealth, query } from './api'
import { ApiKeyModal } from './components/ApiKeyModal'
import { CorpusRail } from './components/CorpusRail'
import { Conversation } from './components/Conversation'
import { DocumentModal } from './components/DocumentModal'
import { IngestModal } from './components/IngestModal'
import { Viewer } from './components/Viewer'
import type {
  CorpusResponse,
  HealthResponse,
  IngestResponse,
  QueryResponse,
  Region,
  Turn,
} from './types'

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [corpus, setCorpus] = useState<CorpusResponse | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [viewer, setViewer] = useState<QueryResponse | null>(null)
  const [asking, setAsking] = useState(false)
  const [ingestOpen, setIngestOpen] = useState(false)
  // The document open in the full-screen viewer, and where to open it. `regions` carries
  // the citation through so the box is still drawn when you arrive from an answer.
  const [doc, setDoc] = useState<{ pdf: string; page: number; regions: Region[] } | null>(null)
  const [toast, setToast] = useState<{ kind: 'ok' | 'err'; msg: string } | null>(null)
  // Set when the server rejects our key (or the absence of one). Every API call routes
  // its 401 here, so an expired key mid-session re-prompts just like a cold load does.
  const [needsKey, setNeedsKey] = useState(false)

  const refreshHealth = useCallback(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: 'degraded', model_loaded: false, qdrant: 'unreachable' }))
  }, [])

  const refreshCorpus = useCallback(() => {
    getCorpus()
      .then(setCorpus)
      .catch((e) => {
        // A gated server answers the very first /corpus with 401 — that is the cold-load
        // path into the key prompt. Anything else: /health surfaces Qdrant connectivity,
        // so leave corpus null.
        if (e instanceof UnauthorizedError) setNeedsKey(true)
      })
  }, [])

  /** Open the document viewer at page 1 - the corpus-rail way in. */
  const openDoc = useCallback((pdf: string) => setDoc({ pdf, page: 1, regions: [] }), [])

  /** Open it at one page, carrying the answer's cited regions so the box is still drawn. */
  const openPage = useCallback(
    (pdf: string, page: number, regions: Region[]) => setDoc({ pdf, page, regions }),
    [],
  )

  /** A 401 raised inside the document viewer has to reach here, or the prompt never opens. */
  const onAuthError = useCallback(() => setNeedsKey(true), [])

  /** Retry the initial loads once a key has been entered. */
  const onKeyEntered = useCallback(() => {
    setNeedsKey(false)
    refreshCorpus()
    refreshHealth()
  }, [refreshCorpus, refreshHealth])

  useEffect(() => {
    refreshCorpus()
    refreshHealth()
  }, [refreshCorpus, refreshHealth])

  useEffect(() => {
    if (!toast) return
    const id = setTimeout(() => setToast(null), 3500)
    return () => clearTimeout(id)
  }, [toast])

  const ask = useCallback(async (question: string) => {
    setAsking(true)
    setViewer(null)
    setTurns((prev) => [...prev, { question, loading: true }])
    try {
      const res = await query(question)
      setTurns((prev) =>
        prev.map((t, i) => (i === prev.length - 1 ? { ...t, loading: false, response: res } : t)),
      )
      setViewer(res)
    } catch (e) {
      if (e instanceof UnauthorizedError) setNeedsKey(true)
      const msg = e instanceof Error ? e.message : 'Query failed.'
      setTurns((prev) =>
        prev.map((t, i) => (i === prev.length - 1 ? { ...t, loading: false, error: msg } : t)),
      )
      setToast({ kind: 'err', msg })
    } finally {
      setAsking(false)
    }
  }, [])

  const onIngestDone = (r: IngestResponse) => {
    const msg = r.indexed_pages === 0
      ? `${r.pdf} was already indexed · unchanged`
      : `${r.pdf} indexed · ${r.indexed_pages} pages`
    setToast({ kind: 'ok', msg })
    refreshCorpus()
    refreshHealth()
  }

  const onDelete = useCallback(async (pdf: string) => {
    try {
      const r = await deleteDocument(pdf)
      setToast({ kind: 'ok', msg: `${r.pdf} removed · ${r.removed_pages} pages` })
      // The viewer renders page images that no longer exist once the document is gone.
      setViewer((v) => (v?.pages.some((p) => p.pdf === pdf) ? null : v))
      setDoc((d) => (d?.pdf === pdf ? null : d))   // same reason: its pages are gone
      refreshCorpus()
    } catch (e) {
      if (e instanceof UnauthorizedError) setNeedsKey(true)
      setToast({ kind: 'err', msg: e instanceof Error ? e.message : 'Delete failed.' })
    }
  }, [refreshCorpus])

  const corpusEmpty = corpus !== null && corpus.total_pages === 0

  return (
    <div className="app">
      <CorpusRail
        corpus={corpus}
        health={health}
        onIngest={() => setIngestOpen(true)}
        onDelete={onDelete}
        onOpen={openDoc}
      />
      <Conversation
        turns={turns}
        onAsk={ask}
        onCite={setViewer}
        asking={asking}
        corpusEmpty={corpusEmpty}
      />
      <Viewer res={viewer} loading={asking} onOpenPage={openPage} />

      {ingestOpen && (
        <IngestModal onClose={() => setIngestOpen(false)} onDone={onIngestDone} />
      )}
      {/* Keyed on (pdf, page): initialPage seeds internal state, so remounting is what
          makes re-opening the same document at a different page actually land there. */}
      {doc && (
        <DocumentModal
          key={`${doc.pdf}@${doc.page}`}
          pdf={doc.pdf}
          initialPage={doc.page}
          regions={doc.regions}
          onClose={() => setDoc(null)}
          onAuthError={onAuthError}
        />
      )}
      {/* Last, so it stacks above the ingest modal if a key expires mid-upload. */}
      {needsKey && <ApiKeyModal onSubmit={onKeyEntered} />}
      {toast && <div className={`toast ${toast.kind}`}>{toast.msg}</div>}
    </div>
  )
}
