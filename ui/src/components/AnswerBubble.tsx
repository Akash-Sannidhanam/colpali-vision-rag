import type { QueryResponse } from '../types'
import { TraceDisclosure } from './TraceDisclosure'

const fmtTok = (t: number) => (t >= 1000 ? `${(t / 1000).toFixed(1)}k` : `${t}`)

/** An answer bubble: the answer text, a citation chip (when found), the summary meta
 *  line, and the expandable per-stage trace. */
export function AnswerBubble({ res, onCite }: { res: QueryResponse; onCite: () => void }) {
  const { answer, citation, pages, meta } = res
  return (
    <div className="msg">
      <div className="bubble-answer">
        <div className={`answer-text${citation.found ? '' : ' muted'}`}>{answer}</div>

        {citation.found && citation.pdf && (
          <button className="cite-chip" onClick={onCite}>
            ◧ {citation.pdf} · p.{citation.page_number} ›
          </button>
        )}

        <div className="meta-line">
          retrieved {meta.retrieve_k} · reranked {pages.length} · {(meta.latency_ms / 1000).toFixed(1)}s ·{' '}
          {fmtTok(meta.total_tokens)} tok · ${meta.est_cost_usd.toFixed(3)}
        </div>

        <TraceDisclosure meta={meta} />
      </div>
    </div>
  )
}
