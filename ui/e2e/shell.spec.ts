import { expect, test } from '@playwright/test'
import { PDF, mockBackend } from './fixtures'

/**
 * The app shell: the ⌘K accelerator and the corpus-integrity warning.
 *
 * Neither is geometry, so this file is not the page-frame guard next door. It is here
 * because both are cross-component wiring that no other layer can reach: vitest covers
 * pure helpers in node, and these two behaviours are a window-level key handler talking to
 * an input three components away, and a server field reaching a rail through App's state.
 * A unit test of either half would have passed for as long as both bugs existed - the
 * shortcut hint was rendered with no handler behind it, and /health's `corpus` field was
 * dropped by the response type before any component could read it.
 */

const SHAPE = { w: 1275, h: 1650 }

// Verbatim in the shape src/server.py:_corpus_status builds it, because the rail decides
// what to show by *excluding* the two sentinel values - a paraphrase would not test that.
const SPLIT = `1 document(s) missing page images (${PDF}) - re-run ingest`

test.describe('the ⌘K accelerator', () => {
  test('focuses the ask box', async ({ page }) => {
    await mockBackend(page, SHAPE)
    await page.goto('/')
    // Wait for /corpus: the input is disabled while the corpus reads empty, and a
    // disabled input cannot take focus.
    await expect(page.getByTitle(`Open ${PDF}`)).toBeVisible()

    const input = page.locator('.askbox input')
    await expect(input).not.toBeFocused()
    await page.keyboard.press('ControlOrMeta+k')
    await expect(input).toBeFocused()
  })

  test('selects what is already typed, so the accelerator replaces it', async ({ page }) => {
    await mockBackend(page, SHAPE)
    await page.goto('/')
    await expect(page.getByTitle(`Open ${PDF}`)).toBeVisible()

    const input = page.locator('.askbox input')
    await input.fill('half a question')
    await page.locator('.convo-head').click()   // focus out
    await page.keyboard.press('ControlOrMeta+k')
    await page.keyboard.type('the real one')
    await expect(input).toHaveValue('the real one')
  })

  test('does not fire while the document viewer holds focus', async ({ page }) => {
    // The guard that matters. DocumentModal runs a Tab trap and restores focus on close,
    // so an accelerator that reached past it would strand the keyboard behind the overlay.
    await mockBackend(page, SHAPE)
    await page.goto('/')
    await page.getByTitle(`Open ${PDF}`).click()
    await expect(page.locator('[role="dialog"]')).toBeVisible()

    await page.keyboard.press('ControlOrMeta+k')
    await expect(page.locator('.askbox input')).not.toBeFocused()
    // Still open, and still the focused region - the key did nothing at all.
    await expect(page.locator('[role="dialog"]')).toBeVisible()
  })
})

test.describe('the corpus-integrity warning', () => {
  test('surfaces the split /health reports', async ({ page }) => {
    // The failure this makes visible is silent by construction: the index outlives its
    // page images, /corpus still lists every document, and every query answers "not
    // found". The rail is the only place a user is told.
    await mockBackend(page, SHAPE, 'ok', SPLIT)
    await page.goto('/')

    const warn = page.locator('.rail-warn')
    await expect(warn).toBeVisible()
    await expect(warn).toContainText(PDF)
    await expect(warn).toContainText('re-run ingest')
  })

  test('stays hidden for a whole corpus and for "unknown"', async ({ page }) => {
    // "unknown" is the server's own placeholder - a 503 body, or a health check that
    // threw - and reporting it as damage would cry wolf on every transient Qdrant blip.
    for (const corpus of ['ok', 'unknown']) {
      await mockBackend(page, SHAPE, 'ok', corpus)
      await page.goto('/')
      await expect(page.getByTitle(`Open ${PDF}`)).toBeVisible()
      await expect(page.locator('.rail-warn')).toHaveCount(0)
    }
  })

  test('stays hidden when the corpus field is absent', async ({ page }) => {
    // The field is optional (older servers, or a startup state before the library exists),
    // and an absent field should not be treated as a problem to report.
    await mockBackend(page, SHAPE, 'ok', null)
    await page.goto('/')
    await expect(page.getByTitle(`Open ${PDF}`)).toBeVisible()
    await expect(page.locator('.rail-warn')).toHaveCount(0)
  })
})
