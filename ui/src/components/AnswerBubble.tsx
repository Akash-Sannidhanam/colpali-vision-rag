import type { QueryResponse } from '../types'
import { TraceDisclosure } from './TraceDisclosure'

const fmtTok = (t: number) => (t >= 1000 ? `${(t / 1000).toFixed(1)}k` : `${t}`)

/** An answer bubble: the answer text, a citation chip (when found), the two confidence
 *  chips, the summary meta line, and the expandable per-stage trace. */
export function AnswerBubble({ res, onCite }: { res: QueryResponse; onCite: () => void }) {
  const { answer, citation, pages, meta } = res
  // The answer-confidence chip is color-coded (high=green, low=red, medium=neutral);
  // the retrieval chip stays neutral and only shows when a page was actually cited.
  const answerConf = citation.confidence
  const retrievalPct =
    meta.retrieval_confidence != null ? Math.round(meta.retrieval_confidence * 100) : null
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
          {retrievalPct != null && (
            <span className="conf-chip" title="How decisively retrieval preferred this page (deterministic, from MaxSim scores).">
              retrieval <b>{retrievalPct}%</b>
            </span>
          )}
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
