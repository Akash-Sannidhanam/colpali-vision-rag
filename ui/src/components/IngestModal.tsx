import { useRef, useState } from 'react'
import { ingestStream } from '../api'
import type { IngestResponse } from '../types'

type Progress = { label: string; page?: number; total?: number }
type Status = {
  phase: 'idle' | 'running' | 'done' | 'error'
  msg?: string
  result?: IngestResponse
  progress?: Progress
  skipped?: boolean // the backend recognised the document and re-embedded nothing
}

/** The ingest modal: drop/choose a PDF, then a render→embed→index run that streams
 *  live per-page progress (SSE) into a progress bar. */
export function IngestModal({
  onClose,
  onDone,
}: {
  onClose: () => void
  onDone: (r: IngestResponse) => void
}) {
  const [file, setFile] = useState<File | null>(null)
  const [drag, setDrag] = useState(false)
  const [status, setStatus] = useState<Status>({ phase: 'idle' })
  const inputRef = useRef<HTMLInputElement>(null)

  const pick = (f: File | null | undefined) => {
    if (f && f.name.toLowerCase().endsWith('.pdf')) {
      setFile(f)
      setStatus({ phase: 'idle' })
    } else if (f) {
      setStatus({ phase: 'error', msg: 'Only .pdf files are accepted.' })
    }
  }

  const run = async () => {
    if (!file) return
    setStatus({ phase: 'running', progress: { label: 'starting…' } })
    let indexed = 0
    let skipped = false
    let pdf = file.name
    try {
      await ingestStream(file, (e) => {
        if (e.phase === 'render') {
          setStatus({ phase: 'running', progress: { label: `rendering ${e.pdf}…` } })
        } else if (e.phase === 'pages') {
          setStatus({ phase: 'running', progress: { label: `${e.total} pages to index`, total: e.total } })
        } else if (e.phase === 'embed') {
          setStatus({
            phase: 'running',
            progress: { label: `embedding page ${e.page} / ${e.total}`, page: e.page, total: e.total },
          })
        } else if (e.phase === 'skip') {
          // Same bytes, same embedding config — the backend re-embedded nothing.
          skipped = true
          setStatus({ phase: 'running', progress: { label: `${e.pdf} already indexed` } })
        } else if (e.phase === 'done') {
          indexed = e.indexed_pages ?? 0
          pdf = e.pdf ?? pdf
        } else if (e.phase === 'error') {
          throw new Error(e.detail ?? 'Ingest failed.')
        }
      })
      const r = { pdf, indexed_pages: indexed }
      setStatus({ phase: 'done', result: r, skipped })
      onDone(r)
    } catch (e) {
      setStatus({ phase: 'error', msg: e instanceof Error ? e.message : 'Ingest failed.' })
    }
  }

  const running = status.phase === 'running'
  const prog = status.progress
  const pct = prog?.total ? Math.round(((prog.page ?? 0) / prog.total) * 100) : null

  return (
    <div className="overlay" onClick={running ? undefined : onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>Ingest a PDF</h2>
        <div className="hint">Pages are indexed as images — no OCR, no text layer.</div>

        {status.phase === 'done' && status.result ? (
          <div className="result-line">
            {status.skipped
              ? `✓ ${status.result.pdf} was already indexed · unchanged`
              : `✓ ${status.result.pdf} indexed · ${status.result.indexed_pages} pages`}
          </div>
        ) : running ? (
          <div className="progress-wrap">
            <div className="progress">
              <div className="spinner" /> {prog?.label ?? `indexing ${file?.name}…`}
            </div>
            {prog?.total ? (
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${pct ?? 0}%` }} />
              </div>
            ) : null}
          </div>
        ) : (
          <div
            className={`dropzone${drag ? ' drag' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault()
              setDrag(true)
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDrag(false)
              pick(e.dataTransfer.files?.[0])
            }}
          >
            <div className="big">{file ? file.name : 'Drop a PDF here or click to choose'}</div>
            <div style={{ font: '500 11px var(--mono)', color: 'var(--t-5)' }}>
              {file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : 'max 50 MB'}
            </div>
            <input
              ref={inputRef}
              type="file"
              accept="application/pdf"
              hidden
              onChange={(e) => pick(e.target.files?.[0])}
            />
          </div>
        )}

        {status.phase === 'error' && (
          <div className="error-line" style={{ marginTop: 12 }}>
            {status.msg}
          </div>
        )}

        <div className="modal-actions">
          {status.phase === 'done' ? (
            <button className="btn primary" onClick={onClose}>
              Done
            </button>
          ) : (
            <>
              <button className="btn" onClick={onClose} disabled={running}>
                Cancel
              </button>
              <button className="btn primary" onClick={run} disabled={!file || running}>
                Index PDF
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
