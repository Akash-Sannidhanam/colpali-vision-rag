import { useState } from 'react'
import { setApiKey } from '../api'

/**
 * Prompts for the deployment's API key after the server answers 401.
 *
 * The key lives in sessionStorage rather than the bundle — a build-time key would ship
 * to every visitor in readable JS. Not dismissible: with no key the app can't load the
 * corpus or ask anything, so an escape hatch would only lead to a dead UI.
 */
export function ApiKeyModal({ onSubmit }: { onSubmit: () => void }) {
  const [value, setValue] = useState('')

  const save = () => {
    const key = value.trim()
    if (!key) return
    setApiKey(key)
    onSubmit()
  }

  return (
    <div className="overlay">
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>API key required</h2>
        <div className="hint">
          This server is gated. Paste the key it was started with — it is kept for this
          browser tab only.
        </div>

        <input
          className="key-input"
          type="password"
          autoFocus
          value={value}
          placeholder="API key"
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && save()}
        />

        <div className="modal-actions">
          <button className="btn primary" onClick={save} disabled={!value.trim()}>
            Unlock
          </button>
        </div>
      </div>
    </div>
  )
}
