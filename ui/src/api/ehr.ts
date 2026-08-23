import { apiFetch } from './http'
import type { CaseResult } from './types'

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? 'https://reassign-guiding-grass.ngrok-free.dev'

export function fetchMyCases(token: string) {
  return apiFetch<CaseResult[]>(`${API_BASE}/api/cases`, token)
}
