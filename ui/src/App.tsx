import { useCallback, useEffect, useState } from 'react'
import { deleteDocument, getCorpus, getHealth, query } from './api'
import { CorpusRail } from './components/CorpusRail'
import { Conversation } from './components/Conversation'
import { IngestModal } from './components/IngestModal'
import { Viewer } from './components/Viewer'
import type {
  CorpusResponse,
  HealthResponse,
  IngestResponse,
  QueryResponse,
  Turn,
} from './types'

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [corpus, setCorpus] = useState<CorpusResponse | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [viewer, setViewer] = useState<QueryResponse | null>(null)
  const [asking, setAsking] = useState(false)
  const [ingestOpen, setIngestOpen] = useState(false)
  const [toast, setToast] = useState<{ kind: 'ok' | 'err'; msg: string } | null>(null)

  const refreshHealth = useCallback(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: 'degraded', model_loaded: false, qdrant: 'unreachable' }))
  }, [])

  const refreshCorpus = useCallback(() => {
    getCorpus()
      .then(setCorpus)
      .catch(() => {
        /* /health surfaces Qdrant connectivity; leave corpus null */
      })
  }, [])

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
      refreshCorpus()
    } catch (e) {
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
      />
      <Conversation
        turns={turns}
        onAsk={ask}
        onCite={setViewer}
        asking={asking}
        corpusEmpty={corpusEmpty}
      />
      <Viewer res={viewer} loading={asking} />

      {ingestOpen && (
        <IngestModal onClose={() => setIngestOpen(false)} onDone={onIngestDone} />
      )}
      {toast && <div className={`toast ${toast.kind}`}>{toast.msg}</div>}
    </div>
  )
}
