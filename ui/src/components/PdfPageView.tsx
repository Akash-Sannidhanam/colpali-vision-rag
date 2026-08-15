import { useEffect, useRef, useState } from 'react'
import type { PDFDocumentProxy } from 'pdfjs-dist'
import { renderTextLayer } from '../pdf'

/**
 * Whether a rejection is pdf.js reporting our own cancellation rather than a real failure.
 * Two names, because the canvas and the text layer each have their own: cancelling a render
 * gives RenderingCancelledException, cancelling a TextLayer rejects with AbortException.
 */
function isCancellation(e: unknown): boolean {
  const name = (e as { name?: string })?.name
  return name === 'RenderingCancelledException' || name === 'AbortException'
}

/**
 * One page of a PDF, rendered at one scale: a canvas, a selectable text layer over it,
 * and whatever overlays the caller passes as children.
 *
 * The frame this returns *is* the page rectangle at the current scale, sized in explicit
 * pixels. That is load-bearing rather than tidy: the citation box is positioned in
 * percentages of this element (`boxToOverlay`), so the frame and the rendered page being
 * the same rectangle is exactly what makes those percentages mean anything. The page-image
 * viewer earned that lesson the expensive way - see e2e/document-viewer.spec.ts.
 *
 * Two rules follow from it and neither is optional:
 *
 *   * Zoom re-renders at a new scale; it never applies a CSS transform. A transform would
 *     scale the canvas while the text layer kept its old geometry, and would put the
 *     frame's layout box and its painted box out of step - which is the same class of bug,
 *     and the reason the e2e suite measures with offsetWidth and not getBoundingClientRect.
 *   * The canvas backing store is `scale x devicePixelRatio` while its CSS size is `scale`.
 *     Rendering at CSS size on a retina display is how you get a crisp vector page that
 *     looks exactly as soft as the PNG it replaced.
 */
export function PdfPageView({
  doc,
  pageNumber,
  scale,
  size,
  onError,
  children,
}: {
  doc: PDFDocumentProxy
  pageNumber: number
  scale: number
  /** The page's CSS size at `scale`, already known to the parent (it computed the fit). */
  size: { w: number; h: number }
  onError: (pageNumber: number) => void
  children?: React.ReactNode
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const textRef = useRef<HTMLDivElement>(null)
  const [painted, setPainted] = useState(false)

  // Reset painted state when page changes (but not when only scale changes)
  useEffect(() => {
    setPainted(false)
  }, [pageNumber])

  useEffect(() => {
    const canvas = canvasRef.current
    const textContainer = textRef.current
    if (!canvas || !textContainer) return

    // Both handles are captured so cleanup can stop work already in flight. Two
    // overlapping render() calls on one canvas interleave their draw commands and tear
    // the page - and a page turn or a zoom click during a render is the normal case here,
    // not an edge one.
    let cancelled = false
    let render: { cancel: () => void } | null = null
    let text: { cancel: () => void } | null = null
    let textEl: HTMLElement | null = null

    ;(async () => {
      try {
        const page = await doc.getPage(pageNumber)
        if (cancelled) return
        const viewport = page.getViewport({ scale })
        // Cap the backing store rather than trusting devicePixelRatio at high zoom: a 6x
        // page on a 3x display is a 100+ megapixel canvas, which browsers silently refuse
        // to allocate and then paint blank.
        const dpr = Math.min(
          window.devicePixelRatio || 1,
          Math.max(1, 4096 / Math.max(viewport.width, viewport.height)),
        )
        // Only the backing store is set here. The canvas's *CSS* size comes from
        // `width/height: 100%` against the frame, and that asymmetry is deliberate: giving
        // the canvas its own pixel size means two roundings of the same number, and the
        // two boxes then disagree by a pixel - which is precisely the frame-is-not-the-page
        // condition the citation overlay cannot survive. e2e caught exactly that.
        canvas.width = Math.floor(viewport.width * dpr)
        canvas.height = Math.floor(viewport.height * dpr)

        // The canvas element, not its 2d context: pdf.js v6 takes the element and manages
        // the context itself, and passing a context instead requires canvas to be null.
        const task = page.render({
          canvas,
          viewport,
          transform: dpr === 1 ? undefined : [dpr, 0, 0, dpr, 0, 0],
        })
        render = task
        await task.promise
        if (cancelled) return
        setPainted(true)

        // The text layer is rendered *after* the canvas and, crucially, in its own try:
        // the page is already on screen by this point, so failing to extract its text must
        // not throw away a canvas that rendered perfectly. Falling back to the PNG there
        // would replace a good render with a worse one to punish a missing convenience.
        try {
          // Each render gets its own element rather than sharing one. A superseded render
          // only learns it was cancelled when its await resolves, which can be after the
          // render that replaced it has already finished - so a shared container has the
          // loser tidying up the winner's spans. A per-render element makes that a no-op
          // on anything visible.
          const el = document.createElement('div')
          el.className = 'textLayer'
          textEl = el
          textContainer.append(el)
          const layer = await renderTextLayer(page, el, viewport)
          if (cancelled) {
            layer.cancel()
            el.remove()
            return
          }
          text = layer
          // Only now drop what the previous render left, so the page never flickers
          // through a state with no text on it.
          for (const old of [...textContainer.children]) if (old !== el) old.remove()
        } catch (e) {
          if (cancelled || isCancellation(e)) return
          textEl?.remove()
          textEl = null
          // Loud, because the failure is otherwise invisible: the page still looks right
          // and only selection and find are quietly gone.
          console.warn(`[pdf] text layer failed for page ${pageNumber}`, e)
        }
      } catch (e) {
        // A cancelled render rejects, and that is this effect's own cleanup talking - not
        // a page that failed. Anything else means we cannot show this page and the caller
        // should fall back to its PNG.
        if (cancelled || isCancellation(e)) return
        console.warn(`[pdf] page ${pageNumber} failed to render; falling back to its image`, e)
        onError(pageNumber)
      }
    })()

    return () => {
      cancelled = true
      render?.cancel()
      // Cancel *and* remove: cancel() empties the element, and leaving an emptied one
      // behind would stack one dead div per zoom click.
      text?.cancel()
      textEl?.remove()
    }
  }, [doc, pageNumber, scale, onError])

  return (
    <div
      className="doc-page pdf"
      style={{
        width: `${size.w}px`,
        height: `${size.h}px`,
        // The text layer's spans are sized in `em`s of this, so it has to be told the
        // scale it was laid out at or every glyph is the wrong size.
        ['--scale-factor' as string]: scale,
      }}
    >
      <canvas ref={canvasRef} />
      {/* A stable host holding one `.textLayer` per render - see the effect above for why
          the layer itself cannot be this element. Above the canvas so text is selectable,
          below the overlays so a citation box is never swallowed by a glyph span. */}
      <div className="text-host" ref={textRef} />
      {painted && children}
    </div>
  )
}
