import type { QueryResponse } from '../types'
import { TraceDisclosure } from './TraceDisclosure'

const fmtTok = (t: number) => (t >= 1000 ? `${(t / 1000).toFixed(1)}k` : `${t}`)

/** An answer bubble: the answer text, a citation chip (when found), the answer-confidence
 *  chip, the summary meta line, and the expandable per-stage trace.
 *
 *  Retrieval decisiveness used to sit here as a second chip reading "retrieval 11%". It
 *  moved into the trace (see `TraceDisclosure`): measured over 73 questions it does
 *  separate a correct top page from a wrong one, but only at AUC 0.629 - too weak to
 *  earn a place beside the answer, and its raw percentage was unreadable besides,
 *  because the value's floor is 1/RETRIEVE_K rather than zero. */
export function AnswerBubble({ res, onCite }: { res: QueryResponse; onCite: () => void }) {
  const { answer, citation, pages, meta } = res
  // Color-coded: high=green, low=red, medium=neutral.
  const answerConf = citation.confidence
  return (
    <div className="msg">
      <div className="bubble-answer">
        <div className={`answer-text${citation.found ? '' : ' muted'}`}>{answer}</div>

        {citation.found && citation.pdf && (
          <button className="cite-chip" onClick={onCite}>
            ◧ {citation.pdf} · p.{citation.page_number} ›
          </button>
        )}

        <div className="conf-row">
          <span
            className={`conf-chip ${answerConf}`}
            title="The model's own self-reported confidence in the answer."
          >
            answer conf <b>{answerConf}</b>
          </span>
        </div>

        <div className="meta-line">
          retrieved {meta.retrieve_k} · reranked {pages.length} · {(meta.latency_ms / 1000).toFixed(1)}s ·{' '}
          {fmtTok(meta.total_tokens)} tok · ${meta.est_cost_usd.toFixed(3)}
        </div>

        <TraceDisclosure meta={meta} />
      </div>
    </div>
  )
}
