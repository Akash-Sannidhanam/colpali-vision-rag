import { Fragment, useState } from 'react'
import type { QueryMeta } from '../types'

const fmtTok = (t: number) => (t >= 1000 ? `${(t / 1000).toFixed(1)}k` : `${t}`)
const fmtCost = (c: number) => `$${c.toFixed(3)}`

/** The "how this was answered" disclosure: per-stage time / tokens / cost from meta.stages. */
export function TraceDisclosure({ meta }: { meta: QueryMeta }) {
  const [open, setOpen] = useState(false)
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
    </div>
  )
}
