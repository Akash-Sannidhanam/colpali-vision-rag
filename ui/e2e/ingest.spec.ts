import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'
import { PAGE_COUNT, PDF, mockBackend, pdf } from './fixtures'

/**
 * The ingest modal's streaming progress.
 *
 * The SSE path is the one flow in this app where the UI is driven by a sequence of server
 * events rather than a single response, and nothing tested it. api.test.ts covers the
 * frame parser in node - including frames split across chunk boundaries, which cannot be
 * staged here because a spec does not control how a response is chunked. What this file
 * covers is the other half: that the parsed events actually move the progress bar, land a
 * result, and refresh the corpus behind the modal.
 */

const SHAPE = { w: 1275, h: 1650 }

const frame = (o: object) => `data: ${JSON.stringify(o)}\n\n`

/** Stub POST /ingest/stream with a scripted run. */
async function stubIngest(page: Page, frames: string[]) {
  await page.route('**/ingest/stream', (route) =>
    route.fulfill({ contentType: 'text/event-stream', body: frames.join('') }),
  )
}

/** Open the modal and attach a real PDF to its file input. */
async function choose(page: Page) {
  await page.getByRole('button', { name: /ingest PDF/ }).click()
  await page.locator('.dropzone input[type=file]').setInputFiles({
    name: 'incoming.pdf',
    mimeType: 'application/pdf',
    buffer: pdf(SHAPE.w, SHAPE.h, 2),
  })
  await expect(page.locator('.dropzone .big')).toContainText('incoming.pdf')
}

test('streams progress, reports the result, and refreshes the corpus', async ({ page }) => {
  await mockBackend(page, SHAPE)
  let corpusFetches = 0
  await page.route('**/corpus', (route) => {
    corpusFetches += 1
    return route.fulfill({
      json: { documents: [{ pdf: PDF, page_count: PAGE_COUNT }], total_pages: PAGE_COUNT, qdrant: 'ok' },
    })
  })
  await stubIngest(page, [
    frame({ phase: 'render', pdf: 'incoming.pdf' }),
    frame({ phase: 'pages', total: 4 }),
    frame({ phase: 'embed', page: 2, total: 4 }),
    frame({ phase: 'embed', page: 4, total: 4 }),
    frame({ phase: 'done', pdf: 'incoming.pdf', indexed_pages: 4 }),
  ])

  await page.goto('/')
  await expect(page.getByTitle(`Open ${PDF}`)).toBeVisible()
  const before = corpusFetches

  await choose(page)
  await page.getByRole('button', { name: 'Index PDF' }).click()

  // The label names the page and the total, which is the same pair the bar's width is
  // computed from - `.progress-fill` is `page / total` as a percentage.
  await expect(page.locator('.progress, .result-line').first()).toBeVisible()
  await expect(page.locator('.result-line')).toContainText('incoming.pdf indexed · 4 pages')
  await expect(page.locator('.toast-region .toast')).toContainText('4 pages')
  // The list behind the modal has to be re-read, or the new document is invisible until
  // a reload.
  expect(corpusFetches).toBe(before + 1)
})

test('an ingest in flight cannot be dismissed', async ({ page }) => {
  // Documented invariant in IngestModal, previously untested: a running ingest holds the
  // GPU, so Cancel is disabled and Esc is ignored. Dismissing it would leave the model
  // busy behind a UI that had forgotten about it.
  //
  // The run is held by delaying the route rather than by pausing mid-stream: route.fulfill
  // delivers a whole body, so a stream that has started has already finished. That is also
  // why the progress bar's width is not asserted here - the reader never observes an
  // intermediate frame. The percentage arithmetic is a pure expression over (page, total)
  // and the label carries the same two numbers, which is what the first test checks.
  await mockBackend(page, SHAPE)
  let release = () => {}
  const held = new Promise<void>((r) => (release = r))
  await page.route('**/ingest/stream', async (route) => {
    await held
    await route.fulfill({
      contentType: 'text/event-stream',
      body: frame({ phase: 'done', pdf: 'incoming.pdf', indexed_pages: 4 }),
    })
  })

  await page.goto('/')
  await expect(page.getByTitle(`Open ${PDF}`)).toBeVisible()
  await choose(page)
  await page.getByRole('button', { name: 'Index PDF' }).click()

  await expect(page.locator('.progress')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Cancel' })).toBeDisabled()

  await page.keyboard.press('Escape')
  await expect(page.locator('.progress')).toBeVisible()   // still there: Esc did nothing

  release()
  await expect(page.locator('.result-line')).toContainText('incoming.pdf')
  // And now that it is over, the modal is dismissible again.
  await expect(page.getByRole('button', { name: 'Done' })).toBeEnabled()
})

test('a terminal error event is shown, and the modal stays open', async ({ page }) => {
  // Closing on failure would take the message with it. The run is over, so Cancel is
  // live again and the visitor can read what happened before dismissing it.
  await mockBackend(page, SHAPE)
  await stubIngest(page, [frame({ phase: 'error', detail: 'Only .pdf files are accepted.' })])

  await page.goto('/')
  await expect(page.getByTitle(`Open ${PDF}`)).toBeVisible()
  await choose(page)
  await page.getByRole('button', { name: 'Index PDF' }).click()

  await expect(page.locator('.error-line')).toContainText('Only .pdf files are accepted.')
  await expect(page.getByRole('button', { name: 'Cancel' })).toBeEnabled()
})

test('an unchanged document is reported as unchanged, not as zero pages', async ({ page }) => {
  // The backend recognised the bytes and re-embedded nothing. "0 pages" would read as a
  // failed ingest.
  await mockBackend(page, SHAPE)
  await stubIngest(page, [
    frame({ phase: 'skip', pdf: 'incoming.pdf' }),
    frame({ phase: 'done', pdf: 'incoming.pdf', indexed_pages: 0 }),
  ])

  await page.goto('/')
  await expect(page.getByTitle(`Open ${PDF}`)).toBeVisible()
  await choose(page)
  await page.getByRole('button', { name: 'Index PDF' }).click()

  await expect(page.locator('.result-line')).toContainText('already indexed · unchanged')
})
