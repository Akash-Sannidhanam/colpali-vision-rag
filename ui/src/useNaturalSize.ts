import { useCallback, useState } from 'react'
import type { SyntheticEvent } from 'react'

/**
 * Track a page image's intrinsic size, which is what gives its frame a definite
 * aspect ratio.
 *
 * Load-bearing, not a convenience. `pageFrameStyle` (lib.ts) turns this into the frame's
 * `aspect-ratio`, and without a definite one the frame is content-sized: `max-height:
 * 100%` on the image resolves to nothing, the image renders at full natural height, and
 * `overflow: hidden` crops the difference away in silence. The visible half is the
 * cropping; the dangerous half is that the citation overlay is positioned in percentages
 * of that same frame, so a cropped frame draws the box against a rectangle the model
 * never measured. That has shipped broken twice, in each of this hook's two callers.
 *
 * Two ways in, because one is not enough. `onLoad` does not fire for an image that was
 * already `complete` at mount - the common case in both callers, since the viewer's page
 * is usually still cached from the candidate rail and the modal preloads its neighbours -
 * so the ref callback covers that, and `onLoad` covers the genuinely-new image.
 *
 * Deliberately never reset between pages or answers: the pages of one document share a
 * size, so holding the last ratio avoids a layout jump on every turn, and either handler
 * corrects it if one differs.
 */
export function useNaturalSize() {
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null)

  const measure = useCallback((el: HTMLImageElement | null) => {
    if (el?.complete && el.naturalWidth) setNatural({ w: el.naturalWidth, h: el.naturalHeight })
  }, [])

  const onLoad = useCallback((e: SyntheticEvent<HTMLImageElement>) => {
    setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })
  }, [])

  // Spread onto the <img>; `natural` goes to pageFrameStyle on the frame around it.
  return { natural, imgProps: { ref: measure, onLoad } }
}
