import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'
import { PAGE_COUNT, PDF, PDF_TEXT, mockBackend } from './fixtures'

/**
 * The geometry guard for the full-screen page frame.
 *
 * This exists because the unit tests cannot see the bug it guards. `pageFrameStyle` and
 * `fitScale` are covered in lib.test.ts, but that only proves a size was *computed*;
 * whether the browser then lays the frame out at it is a layout question, and vitest runs
 * in node. jsdom would not help either - it has no layout engine, so every box measures
 * zero. Only a real browser computes this, which is the entire justification for a
 * Playwright job.
 *
 * What shipped broken once: `max-height: 100%` on the page image resolved against a frame
 * whose own height was content-based and therefore indefinite, so it did nothing. The image
 * rendered at its full natural height and `overflow: hidden` cropped the rest away -
 * measured at 1085px of a 1650px page. The crop is the visible half; the dangerous half is
 * that the citation overlay is positioned in percentages of that same frame, so the box got
 * drawn against a rectangle the model never measured.
 *
 * Hence the assertion is `frame === rendered page` rather than "the page looks right":
 * those two boxes being the same box is exactly what makes a percentage overlay
 * trustworthy. The viewer now renders the real PDF to a canvas and keeps the page image as
 * its fallback, so the claim is made twice - once per path - and it is the same claim.
 */

// Three shapes, to pin both axes of the fit. Portrait and landscape are both bound by the
// stage's height at this viewport; the panorama is wide enough that max-width binds
// instead, which is the branch a height-only fix would pass and still be wrong.
const SHAPES = [
  { name: 'portrait page', w: 1275, h: 1650 },
  { name: 'landscape page', w: 1650, h: 1275 },
  { name: 'panorama page (width-bound)', w: 3000, h: 500 },
]

/** Open the viewer and wait for whichever of the two render paths is expected. */
async function openViewer(page: Page, expecting: 'pdf' | 'image') {
  await page.goto('/')
  await page.getByTitle(`Open ${PDF}`).click()
  if (expecting === 'pdf') {
    // The canvas exists before it is painted, so wait on its backing store having been
    // sized - that happens in the same tick as the render call.
    await expect(page.locator('.doc-page.pdf canvas')).toBeVisible()
    await page.waitForFunction(() => {
      const c = document.querySelector('.doc-page.pdf canvas') as HTMLCanvasElement | null
      return !!c && c.width > 0 && c.height > 0
    })
  } else {
    await expect(page.locator('.doc-page img')).toBeVisible()
    await page.waitForFunction(() => {
      const img = document.querySelector('.doc-page img') as HTMLImageElement | null
      return !!img && img.complete && img.naturalWidth > 0
    })
  }
}

/** Layout geometry, read transform-free. */
async function readGeometry(page: Page) {
  return page.evaluate(() => {
    const frame = document.querySelector('.doc-page') as HTMLElement
    // Whichever path rendered: the canvas of the real PDF, or the page image.
    const inner = frame.querySelector('canvas, img') as HTMLElement
    // .doc-scroll, not .doc-stage: the scroller carries the padding and is the box a page
    // is fitted into. The stage around it exists only to keep the page arrows off the pan.
    const stage = document.querySelector('.doc-scroll') as HTMLElement
    const cs = getComputedStyle(stage)
    const pad = (side: string) => parseFloat(cs.getPropertyValue(`padding-${side}`))
    return {
      // offsetWidth/Height and NOT getBoundingClientRect: the frame carries a `box-in`
      // entry animation that scales it 1.04 -> 1, and getBoundingClientRect includes
      // transforms. Measuring through it reports a phantom overflow that tracks how far
      // the animation has progressed - which cost an hour once already.
      frame: { w: frame.offsetWidth, h: frame.offsetHeight },
      inner: { w: inner.offsetWidth, h: inner.offsetHeight },
      // How much of the page the frame is actually showing.
      visible: { w: frame.clientWidth, h: frame.clientHeight },
      rendered: { w: inner.scrollWidth, h: inner.scrollHeight },
      stage: {
        w: stage.clientWidth - pad('left') - pad('right'),
        h: stage.clientHeight - pad('top') - pad('bottom'),
      },
    }
  })
}

/** The four assertions that must hold on either render path, at fit scale. */
function expectFits(g: Awaited<ReturnType<typeof readGeometry>>, shape: { w: number; h: number }) {
  // 1. The frame and the rendered page are the same box. This is the property the citation
  //    overlay's percentage coordinates depend on, and the one the original bug broke.
  expect(g.frame).toEqual(g.inner)

  // 2. Nothing is clipped: the rendered page does not exceed what the frame shows.
  //    `overflow: hidden` means a violation here is silent in a screenshot.
  expect(g.rendered.h).toBeLessThanOrEqual(g.visible.h)
  expect(g.rendered.w).toBeLessThanOrEqual(g.visible.w)

  // 3. The whole frame fits inside the stage, so nothing bleeds into the header or the
  //    filmstrip.
  expect(g.frame.h).toBeLessThanOrEqual(g.stage.h)
  expect(g.frame.w).toBeLessThanOrEqual(g.stage.w)

  // 4. The page is not distorted. Within a percent, because the fit rounds.
  const want = shape.w / shape.h
  expect(Math.abs(g.frame.w / g.frame.h - want)).toBeLessThan(want * 0.01)

  // 5. It actually fills the stage on its binding axis - a frame that collapsed to a few
  //    pixels would satisfy every assertion above.
  expect(Math.max(g.frame.h / g.stage.h, g.frame.w / g.stage.w)).toBeGreaterThan(0.95)
}

for (const shape of SHAPES) {
  test(`the rendered PDF fits its ${shape.name} without cropping it`, async ({ page }) => {
    await mockBackend(page, shape)
    await openViewer(page, 'pdf')
    expectFits(await readGeometry(page), shape)
  })
}

// The page-image path is not dead code: `has_original: false` is a state the corpus really
// reaches, and every guarantee it had before the PDF renderer existed still has to hold.
// Both routes into it are covered, because one is a decision taken from the manifest and
// the other is a fetch that failed.
for (const source of ['missing', 'error'] as const) {
  test(`falls back to the page image when the source PDF is ${source}`, async ({ page }) => {
    await mockBackend(page, SHAPES[0], source)
    await openViewer(page, 'image')
    expectFits(await readGeometry(page), SHAPES[0])
    // And says so, rather than letting a raster quietly pass for the document.
    await expect(page.locator('.doc-note')).toHaveText('page image')
  })
}

test('the citation overlay lands on the same rectangle the page is drawn in', async ({ page }) => {
  // The overlay is `top/left/width/height` in percentages of `.doc-page`. That is only
  // meaningful while the frame *is* the rendered page, so this asserts the consequence
  // directly: a box the model reported over the middle of the page must land over the
  // middle of the rendered page, not over some crop of it.
  //
  // Repeated after a zoom, which is the failure mode zoom introduces. Zoom re-renders at a
  // new scale rather than applying a CSS transform precisely so that this stays true; a
  // transform would leave the frame's layout box at the old size while the paint moved.
  await mockBackend(page, { w: 1275, h: 1650 })
  await openViewer(page, 'pdf')

  const probe = () =>
    page.evaluate(() => {
      const frame = document.querySelector('.doc-page') as HTMLElement
      const canvas = frame.querySelector('canvas') as HTMLCanvasElement
      // Inject an overlay styled exactly the way boxToOverlay styles a real one: the
      // half-height, half-width centre of the page.
      const el = document.createElement('div')
      el.className = 'box-overlay'
      Object.assign(el.style, { top: '25%', left: '25%', width: '50%', height: '50%' })
      frame.appendChild(el)
      // offsetTop/Left/Width/Height, for the same reason readGeometry uses them:
      // .box-overlay carries its own `box-in` entry animation, and a rect read mid-flight
      // is scaled 1.04 about the overlay's centre - which lands its top edge 1% of the page
      // high. `.doc-page` is position:relative and unpadded, so it is the offsetParent and
      // these are already fractions of the page. The canvas fills it, so dividing by the
      // canvas is the same number - and asserting against the canvas is the claim the test
      // is making.
      const geom = {
        top: el.offsetTop / canvas.offsetHeight,
        left: el.offsetLeft / canvas.offsetWidth,
        height: el.offsetHeight / canvas.offsetHeight,
        width: el.offsetWidth / canvas.offsetWidth,
      }
      el.remove()
      return geom
    })

  const atFit = await probe()
  expect(atFit.top).toBeCloseTo(0.25, 2)
  expect(atFit.left).toBeCloseTo(0.25, 2)
  expect(atFit.height).toBeCloseTo(0.5, 2)
  expect(atFit.width).toBeCloseTo(0.5, 2)

  // Now zoom in twice and make the same claim about the bigger page.
  const before = (await readGeometry(page)).frame
  await page.getByTitle('Zoom in (+)').click()
  await page.getByTitle('Zoom in (+)').click()
  await page.waitForFunction(
    (w) => (document.querySelector('.doc-page') as HTMLElement).offsetWidth > w * 1.4,
    before.w,
  )
  const zoomed = await readGeometry(page)
  expect(zoomed.frame).toEqual(zoomed.inner) // the invariant, at a scale the stage cannot hold
  expect(zoomed.frame.w).toBeGreaterThan(before.w)

  // Both axes, and the ratio. A width-only assertion passes while the page is being
  // stretched: `.doc-page` carries `max-width/max-height: 100%` for the image path, and
  // left on the PDF frame they clamp the zoomed page against the stage on one axis only.
  // Measured before the fix: a 1275x1650 page rendered 810x577. Nothing about that is
  // visible in a screenshot of running text.
  expect(zoomed.frame.h).toBeGreaterThan(before.h)
  const want = 1275 / 1650
  expect(Math.abs(zoomed.frame.w / zoomed.frame.h - want)).toBeLessThan(want * 0.01)

  // And it really does overflow the stage, which is what makes it pannable. A clamped
  // page reports no overflow and silently has nothing to scroll.
  const overflow = await page.evaluate(() => {
    const s = document.querySelector('.doc-scroll') as HTMLElement
    s.scrollLeft = 0
    s.scrollTop = 0
    const frame = document.querySelector('.doc-page') as HTMLElement
    return {
      y: s.scrollHeight > s.clientHeight,
      // With the scroller at its origin, the page's own start edges must be at or after
      // the content origin. A container-centred flex item puts them *before* it, where no
      // amount of scrolling reaches - measured at offsetLeft -510 before this was fixed.
      // offsetLeft/Top rather than a rect, because .doc-page animates in with a transform.
      startEdgesReachable: frame.offsetLeft >= 0 && frame.offsetTop >= 0,
    }
  })
  expect(overflow.y).toBe(true)
  expect(overflow.startEdgesReachable).toBe(true)

  const atZoom = await probe()
  expect(atZoom.top).toBeCloseTo(0.25, 2)
  expect(atZoom.left).toBeCloseTo(0.25, 2)
  expect(atZoom.height).toBeCloseTo(0.5, 2)
  expect(atZoom.width).toBeCloseTo(0.5, 2)
})

test('the page text is really there, positioned over the page', async ({ page }) => {
  // The point of rendering the document rather than its picture: text that can be selected,
  // copied and found. A canvas alone would pass every geometry assertion above and still be
  // a picture.
  await mockBackend(page, { w: 1275, h: 1650 })
  await openViewer(page, 'pdf')

  const layer = page.locator('.doc-page.pdf .textLayer')
  await expect(layer).toContainText(`${PDF_TEXT} 1`)

  // Positioned inside the page frame, not stacked at its origin - the text layer is only
  // useful if its spans sit over the glyphs they belong to.
  const span = await page.evaluate(() => {
    const el = document.querySelector('.textLayer span') as HTMLElement
    const frame = document.querySelector('.doc-page') as HTMLElement
    return {
      w: el.offsetWidth,
      h: el.offsetHeight,
      insideX: el.offsetLeft >= 0 && el.offsetLeft < frame.offsetWidth,
      insideY: el.offsetTop >= 0 && el.offsetTop < frame.offsetHeight,
    }
  })
  expect(span.w).toBeGreaterThan(0)
  expect(span.h).toBeGreaterThan(0)
  expect(span.insideX).toBe(true)
  expect(span.insideY).toBe(true)
})

test('the text survives zooming faster than it can render', async ({ page }) => {
  // A superseded render only learns it was cancelled when its await resolves, which can be
  // after the render that replaced it has finished. `TextLayer.cancel()` empties its
  // container, so while every render shared one container a stale cancel wiped the live
  // page's spans - a perfectly painted canvas with no selectable text on it, reachable by
  // nothing more exotic than clicking + six times. Exactly the failure this whole viewer
  // exists to avoid: it looks right and isn't.
  await mockBackend(page, { w: 1275, h: 1650 })
  await openViewer(page, 'pdf')

  const zoomIn = page.getByTitle('Zoom in (+)')
  for (let i = 0; i < 6; i++) await zoomIn.click()

  await expect(page.locator('.doc-page.pdf .textLayer')).toContainText(`${PDF_TEXT} 1`)
  // And exactly one layer survives - the losers must be removed, not merely emptied.
  await expect(page.locator('.doc-page.pdf .textLayer')).toHaveCount(1)
})

test('paging through the document keeps rendering the real pages', async ({ page }) => {
  // Guards the render-cancellation path: a page turn during a render must leave the new
  // page painted, not the old one and not a torn mix of the two.
  await mockBackend(page, { w: 1275, h: 1650 })
  await openViewer(page, 'pdf')

  for (let n = 2; n <= PAGE_COUNT; n++) {
    await page.getByLabel('Next page').click()
    await expect(page.locator('.doc-count')).toHaveText(`p.${n} / ${PAGE_COUNT}`)
    await expect(page.locator('.doc-page.pdf .textLayer')).toContainText(`${PDF_TEXT} ${n}`)
  }
  // Still the PDF, and still the same box as its frame, after three renders on one canvas.
  const g = await readGeometry(page)
  expect(g.frame).toEqual(g.inner)
})
