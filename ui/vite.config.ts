import { createRequire } from 'node:module'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, join, posix, sep } from 'node:path'
import type { Plugin } from 'vite'
import { configDefaults, defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// pdf.js loads some of its data at runtime rather than bundling it: WebAssembly image
// codecs and a colour-management module, the base-14 font data for PDFs that reference
// Helvetica/Times without embedding them, the CJK CMaps, and an ICC profile. It builds
// those URLs by concatenation - `${wasmUrl}openjpeg.wasm` - so they need their *exact*
// filenames, which rules out Vite's normal hashed-asset pipeline and `?url` imports.
//
// They must also land under `assets/`. The API serves the built UI by mounting only
// /assets plus an explicit GET / (see `_mount_ui` in src/server.py, which is deliberately
// not a catch-all StaticFiles mount - that would shadow the router's 404s). Anything
// emitted at the dist root therefore works under `vite dev` and 404s in the container.
//
// Hence: copied verbatim into assets/pdfjs/<dir>/ at build time, and served from
// node_modules by a dev middleware so both modes resolve the same URLs.
const PDFJS_DIRS = ['wasm', 'standard_fonts', 'cmaps', 'iccs']
const PDFJS_BASE = 'assets/pdfjs'

const MIME: Record<string, string> = {
  '.wasm': 'application/wasm', // instantiateStreaming rejects anything else
  '.js': 'text/javascript',
  '.bcmap': 'application/octet-stream',
  '.pfb': 'application/octet-stream',
  '.ttf': 'font/ttf',
  '.icc': 'application/vnd.iccprofile',
}

/** Every file under `dir`, as paths relative to it, with `/` separators. */
function filesUnder(dir: string, prefix = ''): string[] {
  return readdirSync(dir).flatMap((name) => {
    const abs = join(dir, name)
    const rel = prefix ? posix.join(prefix, name) : name
    return statSync(abs).isDirectory() ? filesUnder(abs, rel) : [rel]
  })
}

function pdfjsAssets(): Plugin {
  const pkg = dirname(createRequire(import.meta.url).resolve('pdfjs-dist/package.json'))
  const assets = PDFJS_DIRS.flatMap((d) =>
    filesUnder(join(pkg, d)).map((rel) => ({ url: `${d}/${rel}`, abs: join(pkg, d, rel.split('/').join(sep)) })),
  )
  return {
    name: 'pdfjs-runtime-assets',
    // Dev only: `vite build` never calls this, and the emit below never runs in dev.
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = req.url?.split('?')[0] ?? ''
        const hit = path.startsWith(`/${PDFJS_BASE}/`)
          ? assets.find((a) => a.url === path.slice(PDFJS_BASE.length + 2))
          : undefined
        if (!hit) return next()
        const ext = hit.url.slice(hit.url.lastIndexOf('.'))
        res.setHeader('Content-Type', MIME[ext] ?? 'application/octet-stream')
        res.end(readFileSync(hit.abs))
      })
    },
    generateBundle() {
      // An explicit fileName means no content hash, which is the whole point: pdf.js
      // asks for these by name and cannot be told a hashed one.
      for (const a of assets) {
        this.emitFile({
          type: 'asset',
          fileName: `${PDFJS_BASE}/${a.url}`,
          source: readFileSync(a.abs),
        })
      }
    },
  }
}

// The UI runs on its own dev server and talks to the FastAPI backend (default
// http://127.0.0.1:8000) over CORS. Override the target with VITE_API_BASE.
export default defineConfig({
  plugins: [react(), pdfjsAssets()],
  server: { port: 5173 },
  test: {
    // Playwright owns e2e/. The two runners share the `*.spec.ts` convention but not a
    // runtime, so without this vitest picks up the browser specs and dies on the
    // @playwright/test import.
    exclude: [...configDefaults.exclude, 'e2e/**'],
  },
})
