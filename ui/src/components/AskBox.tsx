import { useEffect, useRef, useState } from 'react'

// The accelerator is Cmd on Apple hardware and Ctrl everywhere else. The *handler*
// (App.tsx) accepts either key on every platform - only this label has to pick one,
// and a chip that names a key the visitor doesn't have is the bug it used to be.
const IS_MAC = typeof navigator !== 'undefined' && /Mac/i.test(navigator.userAgent)
export const ASK_HINT = IS_MAC ? '⌘K' : 'Ctrl K'

/** The question input. Disabled while a query is in flight or the corpus is empty. */
export function AskBox({
  onAsk,
  disabled,
  placeholder,
  focusSignal,
}: {
  onAsk: (q: string) => void
  disabled: boolean
  placeholder: string
  /** Incremented by the ⌘K handler in App. Any change focuses and selects the input. */
  focusSignal: number
}) {
  const [value, setValue] = useState('')
  const ref = useRef<HTMLInputElement>(null)

  // `> 0` so the initial render doesn't focus: a gated cold load opens the
  // non-dismissible ApiKeyModal, and stealing focus out of it would strand the visitor.
  useEffect(() => {
    if (focusSignal === 0) return
    ref.current?.focus()
    ref.current?.select()   // so the accelerator replaces a half-typed question
  }, [focusSignal])

  const submit = () => {
    const q = value.trim()
    if (!q || disabled) return
    onAsk(q)
    setValue('')
  }

  return (
    <div className="askbox">
      <div className="askbox-inner">
        <input
          ref={ref}
          value={value}
          disabled={disabled}
          aria-label="Ask a question about your documents"
          placeholder={placeholder}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.nativeEvent.isComposing) submit()
          }}
        />
        <span className="ask-hint" aria-hidden="true">{ASK_HINT}</span>
        <button
          className="ask-submit"
          aria-label="Ask"
          disabled={disabled || !value.trim()}
          onClick={submit}
        >
          ↑
        </button>
      </div>
    </div>
  )
}
