import { useEffect, useRef } from 'react'

export type Pane = 'session' | 'page'

/**
 * The narrow-screen chrome: a hamburger that opens the corpus rail as a drawer, and a
 * Session|Page switcher for the single-column layout.
 *
 * Both controls are always rendered and always wired — `theme.css` decides at which
 * widths they are visible (the bar itself below 1100px, the switcher below 820px). Doing
 * it in CSS rather than from a JS media query is what keeps one source of truth for the
 * breakpoints: a `matchMedia` listener here would be a second copy to drift from.
 *
 * **The drawer is a disclosure, not a modal.** It gets `aria-expanded` + `aria-controls`,
 * Esc, and focus returned to the toggle on close — but no `role="dialog"`, no `aria-modal`
 * and no Tab trap, unlike `DocumentModal`. The reason is that the element it controls is
 * the *same* element that is a static landmark above 1100px, so a role that made it a
 * dialog would have to be applied conditionally on viewport width, which needs exactly the
 * JS media query the CSS-only approach above avoids. The cost is real and accepted: Tab
 * from the last control in the open drawer continues into the panes behind the scrim
 * rather than wrapping. The rail is a short list, and a wrong ARIA role at one width is
 * worse than a missing trap at another.
 */
export function AppBar({
  railId,
  railOpen,
  onToggleRail,
  pane,
  onPane,
}: {
  /** The id of the element the hamburger discloses, for aria-controls. */
  railId: string
  railOpen: boolean
  onToggleRail: (open: boolean) => void
  pane: Pane
  onPane: (p: Pane) => void
}) {
  const toggleRef = useRef<HTMLButtonElement>(null)
  // Whether the drawer was open on the previous render, so focus is only restored on an
  // actual close — not on every render that happens to have it shut.
  const wasOpen = useRef(false)

  // Esc closes, and focus goes back to the hamburger. Without the restore, closing from
  // the keyboard drops focus to <body> and the next Tab restarts at the top of the page.
  useEffect(() => {
    if (!railOpen) {
      if (wasOpen.current) toggleRef.current?.focus()
      wasOpen.current = false
      return
    }
    wasOpen.current = true
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onToggleRail(false)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [railOpen, onToggleRail])

  return (
    <div className="appbar">
      <button
        ref={toggleRef}
        className="icon-btn"
        aria-label="Corpus"
        aria-expanded={railOpen}
        aria-controls={railId}
        title="Corpus"
        onClick={() => onToggleRail(!railOpen)}
      >
        ☰
      </button>
      {/* The brand lives in the rail, which is off-canvas at these widths. */}
      <span className="appbar-brand">Vision RAG</span>

      {/* aria-pressed rather than a radiogroup: these are two toggle buttons showing which
          pane is on screen, and arrow-key roving focus would be a surprising amount of
          behaviour for a two-item control that Tab already reaches. */}
      <div className="switch" role="group" aria-label="Visible pane">
        <button aria-pressed={pane === 'session'} onClick={() => onPane('session')}>
          Session
        </button>
        <button aria-pressed={pane === 'page'} onClick={() => onPane('page')}>
          Page
        </button>
      </div>
    </div>
  )
}
