import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'
import { PAGE_COUNT, PDF, mockBackend } from './fixtures'

/**
 * The corpus rail's four states and the delete flow.
 *
 * This is App.tsx's state machine, which no other layer reaches: vitest covers the pure
 * helpers and api.test.ts covers the client, but "a failed /corpus renders an error with a
 * working retry" is a fact about three components and a fetch, and only a browser has all
 * three at once.
 *
 * Two of the states here did not exist until this pass. `corpus === null` rendered as
 * nothing, which is indistinguishable from an empty corpus; and any non-401 failure - a
 * 503 from an unreachable Qdrant, a 429 - left it null *forever*, with no message and no
 * way to try again.
 *
 * Routes registered after mockBackend win: Playwright matches handlers most-recent-first.
 * That is why these specs override `**\/corpus` locally instead of mockBackend growing an
 * option per failure mode.
 */

const SHAPE = { w: 1275, h: 1650 }

const corpusBody = {
  documents: [{ pdf: PDF, page_count: PAGE_COUNT }],
  total_pages: PAGE_COUNT,
  qdrant: 'ok',
}

test.describe('while the corpus is loading', () => {
  test('shows skeleton rows, not an empty corpus', async ({ page }) => {
    await mockBackend(page, SHAPE)
    let release = () => {}
    const held = new Promise<void>((r) => (release = r))
    await page.route('**/corpus', async (route) => {
      await held
      await route.fulfill({ json: corpusBody })
    })

    await page.goto('/')
    await expect(page.locator('.doc-skeleton')).toHaveCount(3)
    // The distinction that matters: a visitor must not read "loading" as "you have
    // nothing indexed", which is what a blank list said - and what "corpus · 0 pages"
    // went on saying one line higher up until the count learned to wait too.
    await expect(page.getByText('No documents yet.')).toHaveCount(0)
    await expect(page.locator('.rail .section-label')).not.toContainText('0 pages')

    release()
    await expect(page.getByTitle(`Open ${PDF}`)).toBeVisible()
    await expect(page.locator('.doc-skeleton')).toHaveCount(0)
  })
})

test.describe('when /corpus fails', () => {
  test('shows the error and a retry that actually retries', async ({ page }) => {
    await mockBackend(page, SHAPE)
    let attempts = 0
    await page.route('**/corpus', (route) => {
      attempts += 1
      return attempts === 1
        ? route.fulfill({ status: 503, json: { detail: 'Qdrant is unreachable.' } })
        : route.fulfill({ json: corpusBody })
    })

    await page.goto('/')
    const err = page.locator('.rail-error')
    await expect(err).toBeVisible()
    await expect(err).toContainText('Qdrant is unreachable.')

    // Clicking it has to re-fetch, not just clear the message - the whole point of the
    // button is that the failure was transient.
    await page.locator('.rail-retry').click()
    await expect(page.getByTitle(`Open ${PDF}`)).toBeVisible()
    await expect(err).toHaveCount(0)
    expect(attempts).toBe(2)
  })

  test('names a rate limit, and does not offer the key prompt', async ({ page }) => {
    // A 429 is not an auth problem, and prompting for a key would send the visitor
    // looking for a credential they already have.
    await mockBackend(page, SHAPE)
    await page.route('**/corpus', (route) =>
      route.fulfill({ status: 429, headers: { 'Retry-After': '30' }, json: { detail: 'slow down' } }),
    )

    await page.goto('/')
    await expect(page.locator('.rail-error')).toContainText('Rate limited')
    // And the count says so rather than claiming an empty corpus.
    await expect(page.locator('.rail .section-label')).toContainText('unavailable')
    await expect(page.locator('.rail-error')).toContainText('30s')
    await expect(page.locator('.modal h2')).toHaveCount(0)
  })

  test('a 401 opens the key prompt instead, and unlocking loads the corpus', async ({ page }) => {
    // The cold-load path on a gated server. The modal is the message, so there is no
    // error line as well.
    await mockBackend(page, SHAPE)
    let gated = true
    await page.route('**/corpus', (route) =>
      gated
        ? route.fulfill({ status: 401, json: { detail: 'Missing API key.' } })
        : route.fulfill({ json: corpusBody }),
    )

    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'API key required' })).toBeVisible()
    await expect(page.locator('.rail-error')).toHaveCount(0)

    gated = false
    await page.locator('.key-input').fill('a-real-key')
    await page.getByRole('button', { name: 'Unlock' }).click()
    await expect(page.getByTitle(`Open ${PDF}`)).toBeVisible()
    await expect(page.getByRole('heading', { name: 'API key required' })).toHaveCount(0)
  })
})

test.describe('removing a document', () => {
  /** Stub DELETE /corpus/<pdf>. `**\/corpus\/*` cannot match a nested /pages or /file. */
  async function stubDelete(page: Page) {
    await page.route('**/corpus/*', (route) =>
      route.request().method() === 'DELETE'
        ? route.fulfill({ json: { pdf: PDF, removed_pages: PAGE_COUNT } })
        : route.fallback(),
    )
  }

  test('confirms inline, then reports through the live region', async ({ page }) => {
    await mockBackend(page, SHAPE)
    await stubDelete(page)
    await page.goto('/')
    await expect(page.getByTitle(`Open ${PDF}`)).toBeVisible()

    await page.getByRole('button', { name: `Remove ${PDF}` }).click()
    await expect(page.getByRole('group', { name: `Remove ${PDF}?` })).toBeVisible()
    await page.getByRole('button', { name: 'yes' }).click()

    // Inside the always-mounted region, which is what makes it announced at all.
    await expect(page.locator('.toast-region .toast')).toContainText('removed')
  })

  test('moves focus onto the confirm, and back out of it', async ({ page }) => {
    // The bug this guards: confirming unmounts the ✕ that was focused, so focus fell to
    // <body> and the next Tab restarted at the top of the page - a keyboard user was
    // thrown out of the row mid-decision.
    await mockBackend(page, SHAPE)
    await stubDelete(page)
    await page.goto('/')
    await expect(page.getByTitle(`Open ${PDF}`)).toBeVisible()

    const remove = page.getByRole('button', { name: `Remove ${PDF}` })
    await remove.click()
    await expect(page.getByRole('button', { name: 'yes' })).toBeFocused()

    await page.getByRole('button', { name: 'no' }).click()
    await expect(remove).toBeFocused()
  })
})
