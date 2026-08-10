import { expect, test } from '@playwright/test'
import { PDF, mockBackend } from './fixtures'

/**
 * The geometry guard for the full-screen page frame.
 *
 * This exists because the unit tests cannot see the bug it guards. `pageFrameStyle` is
 * covered in lib.test.ts, but that only proves a ratio was *emitted*; whether the browser
 * then sizes the frame to its image is a layout question, and vitest runs in node. jsdom
 * would not help either - it has no layout engine, so every box measures zero. Only a real
 * browser computes this, which is the entire justification for a Playwright job.
 *
 * What shipped broken: `max-height: 100%` on the image resolved against a frame whose own
 * height was content-based and therefore indefinite, so it did nothing. The image rendered
 * at its full natural height and `overflow: hidden` cropped the rest away - measured at
 * 1085px of a 1650px page. The crop is the visible half; the dangerous half is that the
 * citation overlay is positioned in percentages of that same frame, so the box got drawn
 * against a rectangle the model never measured.
 *
 * Hence the assertion is `frame === image` rather than "the image looks right": those two
 * boxes being the same box is exactly what makes a percentage overlay trustworthy.
 */

// Three shapes, to pin both axes of the fit. Portrait and landscape are both bound by the
// stage's height at this viewport; the panorama is wide enough that max-width binds
// instead, which is the branch a height-only fix would pass and still be wrong.
const SHAPES = [
  { name: 'portrait page', w: 1275, h: 1650 },
  { name: 'landscape page', w: 1650, h: 1275 },
  { name: 'panorama page (width-bound)', w: 3000, h: 500 },
]

/** Layout geometry, read transform-free. */
async function readGeometry(page: import('@playwright/test').Page) {
  return page.evaluate(() => {
    const frame = document.querySelector('.doc-page') as HTMLElement
    const img = frame.querySelector('img') as HTMLImageElement
    const stage = document.querySelector('.doc-stage') as HTMLElement
    const cs = getComputedStyle(stage)
    const pad = (side: string) => parseFloat(cs.getPropertyValue(`padding-${side}`))
    return {
      // offsetWidth/Height and NOT getBoundingClientRect: the frame carries a `box-in`
      // entry animation that scales it 1.04 -> 1, and getBoundingClientRect includes
      // transforms. Measuring through it reports a phantom overflow that tracks how far
      // the animation has progressed - which cost an hour once already.
      frame: { w: frame.offsetWidth, h: frame.offsetHeight },
      img: { w: img.offsetWidth, h: img.offsetHeight },
      natural: { w: img.naturalWidth, h: img.naturalHeight },
      // How much of the image the frame is actually showing.
      visible: { w: frame.clientWidth, h: frame.clientHeight },
      rendered: { w: img.scrollWidth, h: img.scrollHeight },
      stage: {
        w: stage.clientWidth - pad('left') - pad('right'),
        h: stage.clientHeight - pad('top') - pad('bottom'),
      },
    }
  })
}

for (const shape of SHAPES) {
  test(`the page frame fits its ${shape.name} without cropping it`, async ({ page }) => {
    await mockBackend(page, shape)
    await page.goto('/')

    await page.getByTitle(`Open ${PDF}`).click()
    await expect(page.locator('.doc-page img')).toBeVisible()
    await page.waitForFunction(() => {
      const img = document.querySelector('.doc-page img') as HTMLImageElement | null
      return !!img && img.complete && img.naturalWidth > 0
    })

    const g = await readGeometry(page)

    expect(g.natural).toEqual({ w: shape.w, h: shape.h })

    // 1. The frame and the image are the same box. This is the property the citation
    //    overlay's percentage coordinates depend on, and the one the bug broke.
    expect(g.frame).toEqual(g.img)

    // 2. Nothing is clipped: the image's rendered size does not exceed what the frame
    //    shows. `overflow: hidden` means a violation here is silent in a screenshot.
    expect(g.rendered.h).toBeLessThanOrEqual(g.visible.h)
    expect(g.rendered.w).toBeLessThanOrEqual(g.visible.w)

    // 3. The whole frame fits inside the stage, so nothing bleeds into the header or the
    //    filmstrip.
    expect(g.frame.h).toBeLessThanOrEqual(g.stage.h)
    expect(g.frame.w).toBeLessThanOrEqual(g.stage.w)

    // 4. The page is not distorted. Within a pixel, because the fit rounds.
    const want = shape.w / shape.h
    expect(Math.abs(g.frame.w / g.frame.h - want)).toBeLessThan(want * 0.01)

    // 5. It actually fills the stage on its binding axis - a frame that collapsed to a
    //    few pixels would satisfy every assertion above.
    expect(Math.max(g.frame.h / g.stage.h, g.frame.w / g.stage.w)).toBeGreaterThan(0.95)
  })
}

test('the citation overlay lands on the same rectangle the page is drawn in', async ({ page }) => {
  // The overlay is `top/left/width/height` in percentages of `.doc-page`. That is only
  // meaningful while the frame *is* the image, so this asserts the consequence directly:
  // a box the model reported over the middle of the page must land over the middle of the
  // rendered page, not over some crop of it.
  await mockBackend(page, { w: 1275, h: 1650 })
  await page.goto('/')
  await page.getByTitle(`Open ${PDF}`).click()
  await expect(page.locator('.doc-page img')).toBeVisible()
  await page.waitForFunction(() => {
    const img = document.querySelector('.doc-page img') as HTMLImageElement | null
    return !!img && img.complete && img.naturalWidth > 0
  })

  const probe = await page.evaluate(() => {
    const frame = document.querySelector('.doc-page') as HTMLElement
    const img = frame.querySelector('img') as HTMLImageElement
    // Inject an overlay styled exactly the way boxToOverlay styles a real one: the
    // half-height, half-width centre of the page.
    const el = document.createElement('div')
    el.className = 'box-overlay'
    Object.assign(el.style, { top: '25%', left: '25%', width: '50%', height: '50%' })
    frame.appendChild(el)
    // offsetTop/Left/Width/Height, for the same reason readGeometry uses them: .box-overlay
    // carries its own `box-in` entry animation, and a rect read mid-flight is scaled 1.04
    // about the overlay's centre - which lands its top edge 1% of the page high. `.doc-page`
    // is position:relative and unpadded, so it is the offsetParent and these are already
    // fractions of the page. The image fills it, so dividing by the image is the same
    // number - and asserting against the image is the claim the test is making.
    const geom = {
      top: el.offsetTop / img.offsetHeight,
      left: el.offsetLeft / img.offsetWidth,
      height: el.offsetHeight / img.offsetHeight,
      width: el.offsetWidth / img.offsetWidth,
    }
    el.remove()
    return geom
  })

  expect(probe.top).toBeCloseTo(0.25, 2)
  expect(probe.left).toBeCloseTo(0.25, 2)
  expect(probe.height).toBeCloseTo(0.5, 2)
  expect(probe.width).toBeCloseTo(0.5, 2)
})
