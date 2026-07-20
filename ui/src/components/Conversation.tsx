import { Fragment } from 'react'
import type { QueryResponse, Turn } from '../types'
import { AnswerBubble } from './AnswerBubble'
import { AskBox } from './AskBox'

/** (2) The conversation column: message history + the ask box. */
export function Conversation({
  turns,
  onAsk,
  onCite,
  asking,
  corpusEmpty,
}: {
  turns: Turn[]
  onAsk: (q: string) => void
  onCite: (res: QueryResponse) => void
  asking: boolean
  corpusEmpty: boolean
}) {
  return (
    <div className="convo">
      <div className="convo-head">
        <span className="convo-title">Session</span>
        <span className="convo-sub">no OCR · vision only</span>
      </div>

      <div className="messages">
        {turns.length === 0 && (
          <div className="empty">
            <div className="glyph">⌕</div>
            <div className="sub">
              {corpusEmpty
                ? 'Ingest a PDF to get started — the corpus is empty.'
                : 'Ask anything about your indexed documents.'}
            </div>
          </div>
        )}

        {turns.map((t, i) => (
          <Fragment key={i}>
            <div className="msg user">
              <div className="bubble-user">{t.question}</div>
            </div>
            {t.loading && (
              <div className="msg">
                <div className="bubble-answer">
                  <span className="answer-text muted">
                    thinking
                    <span className="caret" />
                  </span>
                </div>
              </div>
            )}
            {t.error && (
              <div className="msg">
                <div className="bubble-answer">
                  <span className="answer-text" style={{ color: 'var(--red)' }}>
                    {t.error}
                  </span>
                </div>
              </div>
            )}
            {t.response && <AnswerBubble res={t.response} onCite={() => onCite(t.response!)} />}
          </Fragment>
        ))}
      </div>

      <AskBox
        onAsk={onAsk}
        disabled={asking || corpusEmpty}
        placeholder={corpusEmpty ? 'Ingest a document first…' : 'Ask about your documents…'}
      />
    </div>
  )
}
