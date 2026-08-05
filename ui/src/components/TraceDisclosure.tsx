import { Fragment, useState } from 'react'
import { decisivenessVsUniform } from '../lib'
import type { QueryMeta } from '../types'

const fmtTok = (t: number) => (t >= 1000 ? `${(t / 1000).toFixed(1)}k` : `${t}`)
const fmtCost = (c: number) => `$${c.toFixed(3)}`

/** The "how this was answered" disclosure: per-stage time / tokens / cost from
 *  meta.stages, plus retrieval decisiveness.
 *
 *  Decisiveness lives here rather than beside the answer because that is what its
 *  measured strength supports - it separates a right top page from a wrong one at
 *  AUC 0.629 over 73 questions, which is real but weak. Shown as a multiple of an
 *  evenly-spread slate, since the raw share's floor is 1/RETRIEVE_K, not zero. */
export function TraceDisclosure({ meta }: { meta: QueryMeta }) {
  const [open, setOpen] = useState(false)
  const decisiveness = decisivenessVsUniform(meta.retrieval_confidence, meta.retrieve_k)
  return (
    <div className="trace">
      <button className="trace-toggle" onClick={() => setOpen((o) => !o)}>
        <span>{open ? '▾' : '▸'}</span> how this was answered
      </button>
      {open && (
        <div className="trace-table">
          {meta.stages.map((s) => (
            <Fragment key={s.node}>
              <span className="trace-node">{s.node}</span>
              <span className="trace-num">{(s.latency_ms / 1000).toFixed(1)}s</span>
              <span className="trace-num">{s.total_tokens ? `${fmtTok(s.total_tokens)} tok` : '—'}</span>
              <span className="trace-num">{s.gemini_calls ? fmtCost(s.est_cost_usd) : 'local'}</span>
            </Fragment>
          ))}
        </div>
      )}
      {/* Its own line rather than a fifth row of the stage table - it is a property of
          the retrieval result, not a stage that ran. */}
      {open && decisiveness != null && (
        <div
          className="trace-note"
          title="How much more of the retrieval score the cited page held than it would on an evenly-spread slate. 1x means retrieval had no preference at all. Measured over 73 questions this separates a right top page from a wrong one only weakly (AUC 0.629) - read it as a hint, not a reliability score."
        >
          retrieval decisiveness <b>{decisiveness.toFixed(2)}×</b> uniform
        </div>
      )}
    </div>
  )
}
