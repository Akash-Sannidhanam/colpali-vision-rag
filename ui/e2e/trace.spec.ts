import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'
import { PDF, mockBackend, mockQuery, queryResponse } from './fixtures'

/**
 * The LangSmith link in the "how this was answered" disclosure.
 *
 * `meta.trace_url` is null unless the server traces to LangSmith, which is the default,
 * so both states are real: the link must appear only when there is a trace to open, and
 * a null must not render a dead anchor.
 */

const SHAPE = { w: 1275, h: 1650 }
const TRACE = 'https://smith.example/o/1/projects/p/2/r/abc?poll=true'

async function askAndOpenTrace(page: Page, traceUrl: string | null) {
  const res = queryResponse()
  res.meta.trace_url = traceUrl
  await mockBackend(page, SHAPE)
  await mockQuery(page, res)
  await page.goto('/')
  await expect(page.getByTitle(`Open ${PDF}`)).toBeVisible()
  await page.locator('.askbox input').fill('what does the fixture say?')
  await page.locator('.askbox input').press('Enter')
  await page.getByRole('button', { name: /how this was answered/ }).click()
  await expect(page.locator('.trace-table')).toBeVisible()
}

test('links to the trace when the server returns one', async ({ page }) => {
  await askAndOpenTrace(page, TRACE)
  const link = page.getByRole('link', { name: /view trace in LangSmith/ })
  await expect(link).toHaveAttribute('href', TRACE)
  await expect(link).toHaveAttribute('target', '_blank')
  await expect(link).toHaveAttribute('rel', 'noopener noreferrer')
})

test('renders no link when tracing is off', async ({ page }) => {
  await askAndOpenTrace(page, null)
  await expect(page.getByRole('link', { name: /view trace in LangSmith/ })).toHaveCount(0)
})
