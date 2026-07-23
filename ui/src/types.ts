// Mirrors the FastAPI response contract (src/server.py). Keep in sync with the DTOs.

export type Confidence = "high" | "medium" | "low"

export interface ImageRef {
  url: string | null
  data_uri: string | null
}

export interface PageHit {
  index: number // 1-based; matches Citation.source_page
  pdf: string
  page_number: number
  score: number
  image: ImageRef
}

export interface Citation {
  found: boolean
  source_page: number // 1-based index into pages[]; 0 when not found
  box: number[] // [ymin, xmin, ymax, xmax] on a 0-1000 scale; [] when not found
  pdf: string | null
  page_number: number | null
  confidence: Confidence // the model's self-reported answer confidence
}

export interface StageMeta {
  node: string
  latency_ms: number
  prompt_tokens: number
  output_tokens: number
  total_tokens: number
  est_cost_usd: number
  gemini_calls: number
}

export interface QueryMeta {
  request_id: string
  latency_ms: number
  prompt_tokens: number
  output_tokens: number
  total_tokens: number
  est_cost_usd: number
  gemini_calls: number
  retrieve_k: number
  retrieval_confidence: number | null // deterministic softmax share on the cited page; null if none
  stages: StageMeta[]
}

export interface QueryResponse {
  question: string
  answer: string
  citation: Citation
  pages: PageHit[]
  crop: ImageRef | null
  annotated: ImageRef | null
  meta: QueryMeta
}

export interface DocumentInfo {
  pdf: string
  page_count: number
}

export interface CorpusResponse {
  documents: DocumentInfo[]
  total_pages: number
  qdrant: string
}

export interface HealthResponse {
  status: string
  model_loaded: boolean
  qdrant: string
}

export interface IngestResponse {
  pdf: string
  indexed_pages: number
}

// One conversation turn: the question plus its eventual answer (or error/loading).
export interface Turn {
  question: string
  response?: QueryResponse
  error?: string
  loading: boolean
}
