/**
 * The pdf.js seam.
 *
 * Everything that knows pdf.js exists is imported through here, and it is imported
 * lazily: `pdfjs-dist` is ~450 KB of main-thread JS plus a 1.2 MB worker, several times
 * the size of the rest of this bundle. `import('pdfjs-dist')` inside `loadPdf` keeps all
 * of it out of the initial page load - it arrives only when someone opens a document.
 *
 * Why render the PDF at all when every page is already a PNG: the PNGs are the *model's*
 * view of the corpus (150 DPI rasters that `pdf_render` made for ColQwen2), and they do
 * not scale, cannot be selected, and cannot be searched. The reader gets the document.
 */
import type { PDFDocumentProxy, PDFPageProxy, PageViewport, TextLayer } from 'pdfjs-dist'

/** Where the runtime assets pdf.js fetches by name live. See vite.config.ts. */
const ASSET_BASE = `${import.meta.env.BASE_URL}assets/pdfjs/`.replace(/\/{2,}/g, '/')

type PdfjsModule = typeof import('pdfjs-dist')

// One import, one worker configuration, however many documents get opened.
let modulePromise: Promise<PdfjsModule> | null = null

function pdfjs(): Promise<PdfjsModule> {
  modulePromise ??= (async () => {
    try {
      const mod = await import('pdfjs-dist')
      // `?url` so Vite emits the worker as its own asset and hands back the hashed path.
      // Inlining it as a blob would work too but doubles the parse cost on the main thread,
      // which is the one thread this whole design is trying to keep free.
      const workerUrl = (await import('pdfjs-dist/build/pdf.worker.min.mjs?url')).default
      mod.GlobalWorkerOptions.workerSrc = workerUrl
      return mod
    } catch (e) {
      // Clear the cached promise on failure so a future call can retry the import
      modulePromise = null
      throw e
    }
  })()
  return modulePromise
}

/**
 * Parse a PDF's bytes into a document handle.
 *
 * Takes a Blob rather than an ArrayBuffer deliberately: `getDocument({data})` *detaches*
 * the buffer it is handed, so a caller that also wants to offer the file as a download
 * must be able to produce the bytes twice. `Blob.arrayBuffer()` returns a fresh copy each
 * call, so holding the Blob is what makes both possible.
 *
 * The three URL options are not optional in practice. pdf.js fetches this data lazily by
 * exact filename, and leaving them unset degrades silently and unevenly: a PDF that
 * references Helvetica without embedding it renders in a substituted face, an ICC colour
 * space is dropped with nothing but a console warning, and a JBIG2 or JPEG2000 image
 * fails outright. All three are cases where the viewer would show something subtly unlike
 * the page the model actually read.
 *
 * The caller owns the returned document and must hand it to `closePdf` - it holds a worker.
 */
export async function loadPdf(blob: Blob): Promise<PDFDocumentProxy> {
  const mod = await pdfjs()
  return mod.getDocument({
    data: new Uint8Array(await blob.arrayBuffer()),
    wasmUrl: `${ASSET_BASE}wasm/`,
    standardFontDataUrl: `${ASSET_BASE}standard_fonts/`,
    cMapUrl: `${ASSET_BASE}cmaps/`,
    cMapPacked: true,
    iccUrl: `${ASSET_BASE}iccs/`,
  }).promise
}

/**
 * Tear a document down: abort its in-flight requests and destroy its worker.
 *
 * `destroy()` hangs off the *loading task*, not the document, which is easy to get wrong
 * and impossible to notice - the leak is a worker thread per document opened, and nothing
 * about it is visible from the page. Wrapped here so no caller has to know.
 */
export function closePdf(doc: PDFDocumentProxy): void {
  void doc.loadingTask.destroy()
}

/**
 * Render `page`'s text into `container` as positioned, transparent, selectable spans.
 *
 * This is what makes it the document rather than a picture of it: text you can select,
 * copy and find. The spans are laid out in real CSS pixels against `viewport`, which is
 * why zoom in this viewer is a re-render at a new scale and never a CSS transform - a
 * transform would scale the canvas and leave the text layer measuring the old one.
 *
 * Returns the layer so the caller can `cancel()` it; a page turn mid-render otherwise
 * leaves the previous page's spans in the container.
 */
export async function renderTextLayer(
  page: PDFPageProxy,
  container: HTMLElement,
  viewport: PageViewport,
): Promise<TextLayer> {
  const { TextLayer } = await pdfjs()
  const layer = new TextLayer({
    textContentSource: page.streamTextContent(),
    container,
    viewport,
  })
  await layer.render()
  return layer
}
