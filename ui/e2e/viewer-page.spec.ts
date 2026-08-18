import { expect, test } from '@playwright/test'
import type { Locator, Page } from '@playwright/test'
import { CITE_BOX, PDF, mockBackend, mockQuery } from './fixtures'

/**
 * The geometry guard for the *main viewer's* cited page.
 *
 * document-viewer.spec.ts next door makes this same claim about the full-screen dialog,
 * and that guard is why the dialog is correct. This one exists because the claim was
 * never made about the third column, where the identical defect was live: measured on the
 * running app at 1280x860 with a real answer on screen, `.page-frame` laid out at
 * 412x213 around an image of 412x533 - `overflow: hidden` threw 320px of the page away.
 *
 * The mechanism is worth stating, because it is not the same one the dialog had.
 * `.stage` is `flex: 1` in a column that also holds the crop strip and the candidate
 * rail, so the more the answer found, the *less* height the page gets. `.page-frame` then
 * carried `max-height: 100%`, which for a content-sized box with `overflow: hidden` does
 * not resize anything - it clips. The image, capped independently at `max-height: 62vh`,
 * never learned it had to be smaller.
 *
 * The cropping is the visible half. The dangerous half is the same as it was in the
 * dialog: `.box-overlay` is positioned in percentages of this frame, so a frame that is
 * not the page draws the citation somewhere the model never looked - measured at 23px
 * tall where the model's own box was 16% of a 533px page. A confidently wrong visual
 * citation is the one failure this product cannot have.
 *
 * Hence the assertion is `frame === rendered page`, not "the page looks about right".
 */

const SHAPE = { w: 1275, h: 1650 }

/** offsetWidth/offsetHeight, never getBoundingClientRect: `.box-overlay` carries the
 *  `box-in` entry animation and a rect mid-flight reports a phantom ~1% offset. */
function expectRatio(frame: { w: number; h: number }, ratio: number) {
  // Within a pixel, not within a decimal place: offsetWidth/offsetHeight are integers, so
  // the same ratio tolerance means something different at 500px than at 150px, and the
  // small case is the one this file is about.
  expect(Math.abs(frame.w - frame.h * ratio)).toBeLessThanOrEqual(1)
}

async function box(el: Locator): Promise<{ w: number; h: number; top: number; left: number }> {
  return el.evaluate((n: HTMLElement) => ({
    w: n.offsetWidth,
    h: n.offsetHeight,
    top: n.offsetTop,
    left: n.offsetLeft,
  }))
}

async function ask(page: Page, regions?: number) {
  await mockBackend(page, SHAPE)
  await mockQuery(page, regions === undefined ? undefined : (await import('./fixtures')).queryResponse(regions))
  await page.goto('/')
  await expect(page.getByTitle(`Open ${PDF}`)).toBeVisible()
  await page.locator('.askbox input').fill('what does the fixture say?')
  await page.locator('.askbox input').press('Enter')
  await page.locator('.page-frame img').waitFor()
  // The frame is sized from the image's natural size, which is only known once it has
  // decoded. Measuring before that is measuring the fallback.
  await page.waitForFunction(() => {
    const i = document.querySelector<HTMLImageElement>('.page-frame img')
    return !!i?.complete && i.naturalWidth > 0
  })
}

test.describe('the cited page in the viewer column', () => {
  test('the frame is the rendered page, on both axes', async ({ page }) => {
    await ask(page)

    const frame = await box(page.locator('.page-frame'))
    const img = await box(page.locator('.page-frame img'))
    expect(frame.w).toBe(img.w)
    expect(frame.h).toBe(img.h)
    // And it is really the page, not a sliver that happens to match a clipped image:
    // the aspect ratio has to survive too.
    expectRatio(frame, SHAPE.w / SHAPE.h)
  })

  test('the whole page fits the stage - nothing is clipped away', async ({ page }) => {
    await ask(page)

    const stage = page.locator('.stage')
    const inner = await stage.evaluate((n: HTMLElement) => {
      const cs = getComputedStyle(n)
      return {
        w: n.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight),
        h: n.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom),
      }
    })
    const frame = await box(page.locator('.page-frame'))
    // Both axes: a height-only fix passes a width-bound page, and vice versa.
    expect(frame.h).toBeLessThanOrEqual(Math.ceil(inner.h))
    expect(frame.w).toBeLessThanOrEqual(Math.ceil(inner.w))
    // The stage must not have been forced to scroll to hold it either.
    const overflow = await stage.evaluate((n: HTMLElement) => ({
      y: n.scrollHeight - n.clientHeight,
      x: n.scrollWidth - n.clientWidth,
    }))
    expect(overflow.y).toBeLessThanOrEqual(1)
    expect(overflow.x).toBeLessThanOrEqual(1)
  })

  test('the citation box lands where the model put it', async ({ page }) => {
    await ask(page)

    const frame = await box(page.locator('.page-frame'))
    const overlay = await box(page.locator('.box-overlay').first())
    const [ymin, xmin, ymax, xmax] = CITE_BOX

    // The overlay is authored in percentages of the frame, so this only holds when the
    // frame is the page. It is the assertion the 23px-tall box failed.
    expect(overlay.h / frame.h).toBeCloseTo((ymax - ymin) / 1000, 2)
    expect(overlay.w / frame.w).toBeCloseTo((xmax - xmin) / 1000, 2)
    expect(overlay.top / frame.h).toBeCloseTo(ymin / 1000, 2)
    expect(overlay.left / frame.w).toBeCloseTo(xmin / 1000, 2)
  })

  test('survives an answer that found many regions', async ({ page }) => {
    // The regression scales with what the answer found: each region adds a crop to the
    // strip below, and the strip takes its height out of `.stage`. Four is the case that
    // squeezed the page to a ~20px sliver on the running app.
    await ask(page, 4)

    const frame = await box(page.locator('.page-frame'))
    const img = await box(page.locator('.page-frame img'))
    expect(frame.h).toBe(img.h)
    expectRatio(frame, SHAPE.w / SHAPE.h)
    // A frame that has collapsed still "matches" a collapsed image, so pin a floor: the
    // page has to be a page, not a band.
    expect(frame.h).toBeGreaterThan(100)
  })
})
