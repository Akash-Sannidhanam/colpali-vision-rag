import { useRef, useState } from 'react'
import { ingest } from '../api'
import type { IngestResponse } from '../types'

type Status = { phase: 'idle' | 'running' | 'done' | 'error'; msg?: string; result?: IngestResponse }

/** The ingest modal: drop/choose a PDF, then a blocking render→embed→index run with a
 *  simple progress state (live per-page streaming is a deferred enhancement). */
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
    setStatus({ phase: 'running' })
    try {
      const r = await ingest(file)
      setStatus({ phase: 'done', result: r })
      onDone(r)
    } catch (e) {
      setStatus({ phase: 'error', msg: e instanceof Error ? e.message : 'Ingest failed.' })
    }
  }

  const running = status.phase === 'running'

  return (
    <div className="overlay" onClick={running ? undefined : onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>Ingest a PDF</h2>
        <div className="hint">Pages are indexed as images — no OCR, no text layer.</div>

        {status.phase === 'done' && status.result ? (
          <div className="result-line">
            ✓ {status.result.pdf} indexed · {status.result.indexed_pages} pages
          </div>
        ) : running ? (
          <div className="progress">
            <div className="spinner" /> rendering → embedding → indexing {file?.name}… this can take a
            minute.
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
