import { useState } from 'react'

/** The question input. Disabled while a query is in flight or the corpus is empty. */
export function AskBox({
  onAsk,
  disabled,
  placeholder,
}: {
  onAsk: (q: string) => void
  disabled: boolean
  placeholder: string
}) {
  const [value, setValue] = useState('')

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
          value={value}
          disabled={disabled}
          placeholder={placeholder}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.nativeEvent.isComposing) submit()
          }}
        />
        <span className="ask-hint">⌘K</span>
        <button className="ask-submit" disabled={disabled || !value.trim()} onClick={submit}>
          ↑
        </button>
      </div>
    </div>
  )
}
