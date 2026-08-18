import { expect, test } from '@playwright/test'
import type { Locator, Page } from '@playwright/test'
import { PAGE_COUNT, PDF, mockBackend } from './fixtures'

/**
 * The three shell layouts.
 *
 * This is a browser test for the same reason the page-frame guard next door is: the
 * subject is what the layout engine computes. Every claim here is a media query resolving
 * into a grid template, and vitest runs in node while jsdom has no layout engine at all -
 * both would report every box as zero and pass.
 *
 * What it is guarding against is a regression to the state this replaced: `.app` was a
 * hard `grid-template-columns: 220px 1fr 1.15fr` and theme.css contained zero media
 * queries, so below ~1000px the two content columns fell under 400px each and below
 * ~800px the app was unusable. A visitor on a narrow window met that before they met
 * anything else.
 *
 * Measured with offsetWidth/offsetHeight/offsetTop, never getBoundingClientRect: the
 * scrim and several panes carry entry animations, rects include in-flight transforms, and
 * a mid-flight read reports a phantom offset. That artifact has already produced one
 * imaginary bug report in this suite.
 */

const SHAPE = { w: 1275, h: 1650 }

// Named for what each one is testing, not for a device. 1100 and 820 are the breakpoints;
// these sit clear of both so a one-pixel rounding difference cannot decide a test.
const WIDE = { width: 1280, height: 800 }
const TABLET = { width: 1000, height: 800 }
const NARROW = { width: 700, height: 800 }

/** offsetWidth/offsetHeight together - a resize assertion has to pin the axis that is
 *  *not* supposed to bind, or it passes while the other one is being clamped. */
async function box(el: Locator): Promise<{ w: number; h: number }> {
  return el.evaluate((n: HTMLElement) => ({ w: n.offsetWidth, h: n.offsetHeight }))
}

async function boot(page: Page, viewport: { width: number; height: number }) {
  await page.setViewportSize(viewport)
  await mockBackend(page, SHAPE)
  await page.goto('/')
  // The readiness signal has to be a text assertion, not a visibility one: below 1100px
  // every element in the rail is inside a closed drawer, and waiting for one to be
  // *visible* would wait forever at exactly the widths these tests are about. Text
  // assertions do not require visibility, and this string only appears once /corpus has
  // resolved into state.
  await expect(page.locator('.rail .section-label')).toContainText(`${PAGE_COUNT} pages`)
}

test.describe('the wide layout (>= 1101px)', () => {
  test('keeps three columns and shows no app bar', async ({ page }) => {
    await boot(page, WIDE)

    await expect(page.locator('.appbar')).toBeHidden()
    const rail = await box(page.locator('.rail'))
    expect(rail.w).toBe(220)
    expect(rail.h).toBe(WIDE.height) // full height: no app bar row above it

    // The two content panes split the rest in the designed 1 : 1.15 ratio.
    const convo = await box(page.locator('.convo'))
    const viewer = await box(page.locator('.viewer'))
    expect(convo.w + viewer.w + rail.w).toBe(WIDE.width)
    expect(viewer.w / convo.w).toBeCloseTo(1.15, 1)
    expect(convo.h).toBe(WIDE.height)
    expect(viewer.h).toBe(WIDE.height)
  })
})

test.describe('the tablet layout (<= 1100px)', () => {
  test('moves the rail off-canvas and keeps both content panes', async ({ page }) => {
    await boot(page, TABLET)

    await expect(page.locator('.appbar')).toBeVisible()
    // The switcher belongs to the single-column layout only: at this width both panes
    // are on screen, so a control choosing between them would be a no-op.
    await expect(page.locator('.switch')).toBeHidden()
    await expect(page.locator('.rail')).toBeHidden()

    const convo = await box(page.locator('.convo'))
    const viewer = await box(page.locator('.viewer'))
    // Both axes: a rail that was merely narrowed rather than removed from flow would
    // still pass a width-only check on one of these.
    expect(convo.w).toBeGreaterThan(300)
    expect(viewer.w).toBeGreaterThan(300)
    expect(convo.w + viewer.w).toBe(TABLET.width)
    expect(convo.h).toBe(TABLET.height - 48) // the app bar's --appbar-h
    expect(viewer.h).toBe(convo.h)
  })

  test('the hamburger opens the drawer, Esc closes it, focus comes back', async ({ page }) => {
    await boot(page, TABLET)

    const toggle = page.getByRole('button', { name: 'Corpus' })
    const rail = page.locator('.rail')
    await expect(toggle).toHaveAttribute('aria-expanded', 'false')

    await toggle.click()
    await expect(toggle).toHaveAttribute('aria-expanded', 'true')
    await expect(rail).toBeVisible()
    // Actually on screen, not merely painted: the closed state is translated fully out.
    expect(await rail.evaluate((n: HTMLElement) => n.offsetLeft)).toBe(0)

    await page.keyboard.press('Escape')
    await expect(toggle).toHaveAttribute('aria-expanded', 'false')
    await expect(rail).toBeHidden()
    // Restored, because closing from the keyboard otherwise drops focus to <body> and
    // the next Tab restarts at the top of the document.
    await expect(toggle).toBeFocused()
  })

  test('the closed drawer is out of the tab order', async ({ page }) => {
    // A translated-away element still takes focus; only visibility/display removes it.
    // Without this the first Tab past the hamburger disappears into an invisible rail.
    await boot(page, TABLET)
    await expect(page.getByTitle(`Open ${PDF}`)).toBeHidden()
    await expect(page.locator('.rail .ingest-btn')).toBeHidden()
  })

  test('the scrim closes the drawer', async ({ page }) => {
    await boot(page, TABLET)
    await page.getByRole('button', { name: 'Corpus' }).click()
    await expect(page.locator('.rail-scrim')).toBeVisible()

    await page.locator('.rail-scrim').click()
    await expect(page.locator('.rail')).toBeHidden()
    await expect(page.locator('.rail-scrim')).toHaveCount(0)
  })

  test('opening a document from the drawer closes it and stacks the dialog above', async ({
    page,
  }) => {
    await boot(page, TABLET)
    await page.getByRole('button', { name: 'Corpus' }).click()
    await page.getByTitle(`Open ${PDF}`).click()

    await expect(page.locator('[role="dialog"]')).toBeVisible()
    await expect(page.locator('.rail')).toBeHidden()
    await expect(page.locator('.rail-scrim')).toHaveCount(0)
  })
})

test.describe('the narrow layout (<= 820px)', () => {
  test('shows one pane at a time, at full height', async ({ page }) => {
    await boot(page, NARROW)

    await expect(page.locator('.switch')).toBeVisible()
    await expect(page.locator('.convo')).toBeVisible()
    await expect(page.locator('.viewer')).toBeHidden()

    // The whole point of switching rather than stacking: the visible pane gets the entire
    // height below the app bar, not half of it.
    const convo = await box(page.locator('.convo'))
    expect(convo.w).toBe(NARROW.width)
    expect(convo.h).toBe(NARROW.height - 48)
  })

  test('the switcher swaps which pane is on screen', async ({ page }) => {
    await boot(page, NARROW)

    const session = page.getByRole('button', { name: 'Session' })
    const pagePane = page.getByRole('button', { name: 'Page' })
    await expect(session).toHaveAttribute('aria-pressed', 'true')

    await pagePane.click()
    await expect(pagePane).toHaveAttribute('aria-pressed', 'true')
    await expect(session).toHaveAttribute('aria-pressed', 'false')
    await expect(page.locator('.viewer')).toBeVisible()
    await expect(page.locator('.convo')).toBeHidden()

    const viewer = await box(page.locator('.viewer'))
    expect(viewer.w).toBe(NARROW.width)
    expect(viewer.h).toBe(NARROW.height - 48)

    await session.click()
    await expect(page.locator('.convo')).toBeVisible()
    await expect(page.locator('.viewer')).toBeHidden()
  })

  test('the hidden pane keeps its state - it is not unmounted', async ({ page }) => {
    // Why this matters: the viewer caches a fetched MaxSim heatmap and holds the user's
    // sticky heatOn choice, and the conversation holds its scroll position. Unmounting
    // the pane that is off screen would re-fetch a heatmap on every toggle.
    await boot(page, NARROW)
    await page.locator('.askbox input').fill('a half-typed question')

    await page.getByRole('button', { name: 'Page' }).click()
    await page.getByRole('button', { name: 'Session' }).click()

    await expect(page.locator('.askbox input')).toHaveValue('a half-typed question')
  })
})

test.describe('reduced motion', () => {
  test('is honoured, including by the drawer slide', async ({ page }) => {
    // The drawer transition is the one piece of motion here that moves a whole panel
    // across the screen, which is exactly the class of animation the preference exists
    // for - so it ships with the escape hatch rather than after it.
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await boot(page, TABLET)

    const duration = await page
      .locator('.rail')
      .evaluate((n: HTMLElement) => getComputedStyle(n).transitionDuration)
    expect(parseFloat(duration)).toBeLessThan(0.001)
  })
})
